
#ifndef _IMPLICIT_VOXEL_MAP_H
#define _IMPLICIT_VOXEL_MAP_H

#include <array>
#include <atomic>
#include <cstdint>
#include <shared_mutex>
#include <vector>
#include <torch/script.h>
#include "c10/core/Device.h"
#include "preprocess.hpp"
#include "ros/publisher.h"
#include "voxel_map_util.hpp"
#include "p2v_model.h"
#include "common_lib.h"
#include <mutex>

////////////////////////////////////////////////////////////////////////////
//                          Implicit Voxel Grid
////////////////////////////////////////////////////////////////////////////

class ImplicitVoxelGrid
{
public:
    ImplicitVoxelGrid(float _voxel_size, int _max_point_thresh, int _predictable_thresh, int _update_thresh, VOXEL_LOC _pos);

    // config.
    const VOXEL_LOC position;
    const double voxel_size;
    const int max_point_thresh;         // default: XXX. > this value, no more update or feature extraction.
    const int predictable_thresh;       // default: XXX. > this value, feature can be extracted. then in thread, has_feature = true.
    const int update_thresh;            // default: 10. If wat_to_add_points > 10, then add them to `points` (and norm_points).

    // id
    static uint64_t count;
    uint64_t group_id;

    // data
    V3D center;
    std::vector<V3D> points;            // not needed? 
    std::vector<V3D> norm_points;
    std::vector<V3D> tmp_points;        // these points need to be added to the `points` for feature extraction.
    
    // std::mutex mutex;
    std::mutex m_pointadd;          // lock the tmp_points save/read.
    std::mutex m_feat;              // lock the feature save/read.

    // state
    bool is_full = false;
    std::atomic<int> last_update_id{0};

    bool has_feature = false;
    torch::Tensor feature;

    void addPoint(const V3D& p);
    void printInfo(void){
        std::cout << "[ImplicitVoxelGrid info]. " 
            << "id: " << group_id 
            << ", center: [" << center.x() << ", " << center.y() << ", " << center.z() << "] "
            << ", index: [" << position.x << ", " << position.y << ", " << position.z << "] "
            // << ", points size: " << points.size() 
            << std::endl;
    }

    inline void touch(int scan_id){
        last_update_id.store(scan_id, std::memory_order_relaxed);
    }
    inline int lastUpdate() const {
        return last_update_id.load(std::memory_order_relaxed);
    }


};




// ////////////////////////////////////////////////////////////////////////////
// //                          Implicit Voxel Map
// ////////////////////////////////////////////////////////////////////////////


struct VoxelMapState{
    size_t voxels = 0;
    size_t pts_size = 0, pts_cap = 0;
    size_t tmp_size = 0, tmp_cap = 0;
    size_t norm_size = 0, norm_cap = 0;
    size_t feat_size = 0;
    size_t memory_usage_byte = 0;
};



typedef std::unordered_map<VOXEL_LOC, std::shared_ptr<ImplicitVoxelGrid>> implicitUnorderMap;

class ImplicitVoxelMap{
public:
    ImplicitVoxelMap(ros::NodeHandle* nh, int _max_point_thresh, int _predictable_thresh, int _update_thresh, double _voxel_size, const std::string& encoder, const std::string& decoder, torch::Device dev, const int BATCH_SIZE);

    VOXEL_LOC index(const V3D &p);
    void getNeighborIndices(const V3D &p, std::array<VOXEL_LOC, 27> & out_keys);
    void build(const std::vector<V3D>& points);         // build the init map.
    void update(const std::vector<V3D>& points);
    
    int buildResidual(std::vector<p2v>& p2v_list);       // buildResidual seperate into Batch and predict.
    int buildResidualBatch(std::vector<p2v*>& p2v_batch, bool is_remaining=false); 

    void ensure_cpu_buffers(int B, int Nn);

    void getFullPointCloud(PointCloudXYZI& cloud);
    void printMapInfo(void);
    void publishFullMap(void);
    void publishVoxelCube(void);


public:
    ros::NodeHandle* nh_;
    ros::Publisher pub_fullMap_;
    ros::Publisher pub_voxeCube_;
    mutable std::shared_mutex map_mtx_;
    bool is_map_inited;
    implicitUnorderMap implicit_map;
    const double voxel_size;
    const int predictable_thresh;
    const int max_point_thresh;
    const int update_thresh;

    // models
    torch::Device device;
    const int BATCH_SIZE;
    const int Nn = 27;      // neighbor size
    const int K = 40;       // point in voxel
    const int C = 3;        // channel

    // buffers to speed-up
    torch::Tensor feat_host_;        // [B,Nn,64] float32 pinned
    torch::Tensor u_host_;        // [B,21]    float32 pinned
    torch::Tensor exist_mask_host_;  // [B,Nn]    uint8  pinned
    std::vector<bool> valid_b_;      // [B]

    // Using ONNX
    std::unique_ptr<P2V_Encoder_ONNX> p2v_encoder_onnx; // VE-Net in paper
    std::unique_ptr<P2V_Decoder_ONNX> p2v_decoder_onnx; // IR-Net in paper
};

#endif
