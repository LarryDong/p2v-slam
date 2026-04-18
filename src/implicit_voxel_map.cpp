
#include <cassert>
#include <array>
#include <cstdint>
#include <cstring>
#include <omp.h>
#include <ros/time.h>
#include <ros/publisher.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/PointCloud2.h>

#include "implicit_voxel_map.hpp"
#include "ATen/core/TensorBody.h"
#include "c10/core/DeviceType.h"
#include "c10/core/Device.h"
#include "preprocess.hpp"
#include "voxel_map_util.hpp"
#include "common_lib.h"
#include "my_tools.hpp"
#include "p2v_model.h"

#include <pcl/impl/point_types.hpp>
#include <pcl/point_cloud.h>



// ////////////////////////////////////////////////////////////////////////////
// //                          Implicit Voxel Grid
// ////////////////////////////////////////////////////////////////////////////
uint64_t ImplicitVoxelGrid::count = 0;
extern int g_scan_id;
ImplicitVoxelGrid::ImplicitVoxelGrid(float _voxel_size, int _max_point_thresh, int _predictable_thresh, int _update_thresh, VOXEL_LOC _position)
    : position(_position), voxel_size(_voxel_size), max_point_thresh(_max_point_thresh), predictable_thresh(_predictable_thresh), update_thresh(_update_thresh)
{
    is_full = false;
    has_feature = false;
    tmp_points.reserve(update_thresh);
    points.reserve(max_point_thresh);
    norm_points.reserve(max_point_thresh);
    group_id = count++;
    center = V3D(_position.x + 0.5, _position.y + 0.5, _position.z + 0.5) * voxel_size;
}

void ImplicitVoxelGrid::addPoint(const V3D &p){                 
    // only add, no update.
    if (is_full)
        return ;
    if(norm_points.size() > max_point_thresh && has_feature){        // no more update.
        is_full = true;
        // points.resize(0);
        norm_points.resize(0); 
        return ;
    }
    m_pointadd.lock();
    touch(g_scan_id);       // get last update id.
    tmp_points.push_back(p);
    m_pointadd.unlock();
}

////////////////////////////////////////////////////////////////////////////
//                          Implicit Voxel Map
////////////////////////////////////////////////////////////////////////////

// Only on implicit voxel map (iVoxMap) in the SLAM system, which organize a hash table of implicit voxel grids.

ImplicitVoxelMap::ImplicitVoxelMap(ros::NodeHandle* nh, int _max_point_thresh, int _predictable_thresh, int _update_thresh, double _voxel_size, const string& encoder, const string& decoder, torch::Device dev, const int batch_size) 
    : device(dev), voxel_size(_voxel_size), predictable_thresh(_predictable_thresh), max_point_thresh(_max_point_thresh), update_thresh(_update_thresh), 
      BATCH_SIZE(batch_size), nh_(nh)
{
    is_map_inited = false;
    cout << "ImplicitVoxelMap inited. Voxel_size: " << voxel_size 
            << ", update per: " << update_thresh 
            << ", predict after: " << predictable_thresh 
            << ", max size: " << max_point_thresh
            << ", batch size: " << BATCH_SIZE
            << ", encoder onnx: " << encoder 
            << ", decoder onnx: " << decoder
            << endl;

    p2v_encoder_onnx.reset(new P2V_Encoder_ONNX(encoder));
    p2v_decoder_onnx.reset(new P2V_Decoder_ONNX(decoder));    

    pub_fullMap_ = nh_->advertise<sensor_msgs::PointCloud2>("/implicit/full_map", 1000);
    pub_voxeCube_ = nh_->advertise<visualization_msgs::MarkerArray>("/implicit/cube", 1000);

    implicit_map.clear();
}

VOXEL_LOC ImplicitVoxelMap::index(const  V3D &point){
    Eigen::Vector3d idx = (point / voxel_size).array().floor();
    return VOXEL_LOC(static_cast<int64_t>(idx(0)), static_cast<int64_t>(idx(1)), static_cast<int64_t>(idx(2)));
}

// // Not used codes. vector version.
// std::vector<VOXEL_LOC> ImplicitVoxelMap::getNeighborIndices(const V3D &p){
//     vector<VOXEL_LOC> vvks;
//     vvks.reserve(27);
//     VOXEL_LOC idx = index(p);
//     for (int ix = -1; ix <= 1; ix++)
//         for (int iy = -1; iy <= 1; iy++)
//             for (int iz = -1; iz <= 1; iz++){
//                 VOXEL_LOC idx_neigh = VOXEL_LOC(idx.x + ix, idx.y + iy, idx.z + iz);
//                 vvks.push_back(idx_neigh);
//             }
//     return vvks;
// }

// 2026.1.4 array version
inline void ImplicitVoxelMap::getNeighborIndices(const V3D &p, std::array<VOXEL_LOC, 27> &out_keys){
    const VOXEL_LOC idx = index(p);
    int k = 0;
    for (int ix = -1; ix <= 1; ix++)
        for (int iy = -1; iy <= 1; iy++)
            for (int iz = -1; iz <= 1; iz++) {
                out_keys[k++] = VOXEL_LOC(idx.x + ix, idx.y + iy, idx.z + iz);
            }
}

void ImplicitVoxelMap::printMapInfo(void){
    // cout <<"------------ Implicit Voxel Map Info: ------------" << endl;
    // cout <<"voxel size: " << voxel_size << endl;
    // cout <<"predict_thresh: " << predict_thresh << endl;
    // cout <<"max point thresh: " << max_point_thresh << endl;
    // cout <<"Number of voxels: " << implicit_map.size() << endl;

    for(auto item : implicit_map){
        std::shared_ptr<ImplicitVoxelGrid> vg_ptr = item.second;
        int id = vg_ptr->group_id;
        cout << "Voxel : " << vg_ptr->group_id 
            << " center: [" << vg_ptr->center.x() << ", " << vg_ptr->center.y() << ", " << vg_ptr->center.z() << "] "
            << " tmp poins: " << vg_ptr->tmp_points.size() << ", "
            << " has_feature? : " << vg_ptr->has_feature << ", " << endl;
    }
}



void ImplicitVoxelMap::build(const vector<V3D>& pts){
    ROS_INFO("ImplicitVoxelMap building...");
    for (const V3D& p : pts){
        VOXEL_LOC k = index(p);
        auto it = implicit_map.find(k);
        if (it == implicit_map.end()){
            implicit_map[k] = std::make_shared<ImplicitVoxelGrid>(voxel_size, max_point_thresh, predictable_thresh, update_thresh, k);
            // auto c = implicit_map[k]->center;
            // cout << "ImplicitVoxelMap::build. create center @: [" << c.x()<<", "<< c.y() <<", " << c.z() << "], query: " << p.transpose() << endl;
        }
        implicit_map[k]->addPoint(p);
    }
    return;
}


std::mutex m_map;
void ImplicitVoxelMap::update(const std::vector<V3D>& pts){
    // cout << "ImplicitVoxelMap: update ----------- " << endl;
    for (const auto &p : pts) {
        VOXEL_LOC k = index(p);
        std::shared_ptr<ImplicitVoxelGrid> grid;
        {
            std::lock_guard<std::mutex> lk(m_map);
            auto it = implicit_map.find(k);
            if (it == implicit_map.end()) {
                auto sp = std::make_shared<ImplicitVoxelGrid>(voxel_size, max_point_thresh, predictable_thresh, update_thresh, k);
                it = implicit_map.emplace(k, sp).first;
            }
            grid = it->second;
        }
        grid->addPoint(p);
    }
}



void ImplicitVoxelMap::getFullPointCloud(PointCloudXYZI& pc){
    pc.points.clear();
    for(auto iter=implicit_map.begin(); iter!=implicit_map.end(); ++iter){
        auto vg = iter->second;
        vg->m_pointadd.lock();
        // for(const V3D& p : vg->points){          // original points are removed.
        //     pcl::PointXYZINormal pp;
        //     pp.x = p.x();
        //     pp.y = p.y();
        //     pp.z = p.z();
        //     pc.points.push_back(pp);
        // }
        for(const V3D& p : vg->tmp_points){
            pcl::PointXYZINormal pp;
            pp.x = p.x();
            pp.y = p.y();
            pp.z = p.z();
            pc.points.push_back(pp);
        }
        vg->m_pointadd.unlock();
    }
}



void ImplicitVoxelMap::publishFullMap(void){
    PointCloudXYZI pc;
    getFullPointCloud(pc);
    sensor_msgs::PointCloud2 msg = my_tools::pcl2msg(pc);
    pub_fullMap_.publish(msg);
    // cout <<"Full map published"<<endl;
}


void ImplicitVoxelMap::publishVoxelCube(void){
    string frame_id = "camera_init";
    string ns = "voxel_grid";
    auto stamp = ros::Time::now();
    visualization_msgs::MarkerArray array_msg;

    {   // clear
        visualization_msgs::Marker clear;
        clear.header.stamp = stamp;
        clear.ns = ns;
        clear.id = 0;
        clear.action = visualization_msgs::Marker::DELETEALL;
        array_msg.markers.push_back(clear);
    }

    const double s = voxel_size * 0.95;       // make some margin
    int id = 0;

    for (const auto &kv : implicit_map) {
        const VOXEL_LOC &key = kv.first;

        const double cx = kv.second->center.x();
        const double cy = kv.second->center.y();
        const double cz = kv.second->center.z();

        visualization_msgs::Marker m;
        m.header.frame_id = frame_id;
        m.header.stamp = stamp;
        m.ns = ns;
        m.id = ++id;
        m.type = visualization_msgs::Marker::CUBE;
        m.action = visualization_msgs::Marker::ADD;
        m.pose.position.x = cx;
        m.pose.position.y = cy;
        m.pose.position.z = cz;
        m.pose.orientation.x = 0.0;
        m.pose.orientation.y = 0.0;
        m.pose.orientation.z = 0.0;
        m.pose.orientation.w = 1.0;
        m.scale.x = s;
        m.scale.y = s;
        m.scale.z = s;

        // color: red = full, green = predictable, gray = unpredictable
        V3D color(0,0,0);
        bool flag_feature = kv.second->has_feature;
        bool flag_full = kv.second->is_full;
        if(flag_full)
            color = V3D(0.5, 0, 0);
        else if(!flag_full  && flag_feature)
            color = V3D(0, 1, 0);
        else
            color = V3D(0.5, 0.5, 0.5);
        m.color.r = color[0];
        m.color.g = color[1];
        m.color.b = color[2];
        m.color.a = 0.3f;
        m.lifetime = ros::Duration(0.0);
        array_msg.markers.push_back(m);
    }

    pub_voxeCube_.publish(array_msg);
}

int ImplicitVoxelMap::buildResidual(vector<p2v>& p2v_list){
    double t0 = omp_get_wtime();
    if(!is_map_inited)
        return -1;

    int num_query = p2v_list.size();
    if(num_query == 0)
        return 0;

    vector<p2v*> batch_list;
    batch_list.reserve(BATCH_SIZE);
    int good_cnt = 0, bad_cnt = 0;

    for (int i = 0; i < num_query; ++i) {         
        p2v* v = &(p2v_list[i]);
        v->is_valid = false;
        VOXEL_LOC position = index(v->query_world);
        
        auto iter = implicit_map.find(position);
        if (iter != implicit_map.end()) {
            auto vg_ptr = iter->second;
            batch_list.push_back(v);
            // cout << "Batch size: " << BATCH_SIZE << ", list size: " << batch_list.size() << endl;
            if(batch_list.size() == BATCH_SIZE){
                int cnt_valid_p2v = buildResidualBatch(batch_list);
                good_cnt += cnt_valid_p2v;
                bad_cnt += BATCH_SIZE - cnt_valid_p2v;
                batch_list.clear();
            }
        }
        else
            bad_cnt++;
    }

    if(!batch_list.empty()){
        int cnt_valid_p2v = buildResidualBatch(batch_list, true);
        good_cnt += cnt_valid_p2v;
        bad_cnt += batch_list.size() - cnt_valid_p2v;
    }

    double dur = omp_get_wtime() - t0;

    return good_cnt;
}



int ImplicitVoxelMap::buildResidualBatch(std::vector<p2v*>& batch_list, bool is_remaining){
    if (batch_list.empty()) return 0;

    double t0 = omp_get_wtime();
    assert(device == torch::kCPU);
    if(!is_remaining)
        assert(BATCH_SIZE == batch_list.size());
    else
        assert(BATCH_SIZE >= batch_list.size());

    const int B = batch_list.size();       // batch size is not fixed. The last batch maybe not full
    ensure_cpu_buffers(B, Nn);
    
    float* feat_ptr = feat_host_.data_ptr<float>();         // [B,Nn,64]
    float* u_ptr = u_host_.data_ptr<float>();
    uint8_t* mask_ptr = exist_mask_host_.data_ptr<uint8_t>(); // [B,Nn]
    std::memset(feat_ptr, 0, (size_t)B * Nn * 64 * sizeof(float));
    std::memset(mask_ptr, 0, (size_t)B*Nn*sizeof(uint8_t));
    std::fill(valid_b_.begin(), valid_b_.end(), false);

    for (int b = 0; b < B; ++b) {
        p2v* data = batch_list[b];
        V3D query = data->query_world;
        data->is_valid = false;

        VOXEL_LOC center_key = index(query);                // get central voxel hash key
        auto center_iter = implicit_map.find(center_key);
        if (center_iter == implicit_map.end()) {
            data->is_valid = false;
            continue;
        }
        auto center_grid_ptr = center_iter->second;
        if (!center_grid_ptr->has_feature) {                // no feature, continue.
            data->is_valid = false;
            continue;
        }

        // check neighbor states
        std::array<VOXEL_LOC, 27> neighbor_keys;
        getNeighborIndices(query, neighbor_keys);
        bool has_any = false;       // a flag for "If a neighborhood voxel has feature"?

        for (int v = 0; v < Nn; ++v) {
            const int base = b * Nn + v;
            auto itv = implicit_map.find(neighbor_keys[v]);
            if (itv == implicit_map.end() || !itv->second->has_feature)     // no voxel or no feature, continue
                continue;

            auto vg = itv->second;
            vg->m_feat.lock();
            std::memcpy(feat_ptr + base*64, vg->feature.data_ptr<float>(), 64 * sizeof(float));
            vg->m_feat.unlock();
            has_any = true;
            mask_ptr[base] = 1;      // mask_ptr[] must here set to 1. No matter has feature or not. 
        }

        V3D query_local = (query - center_grid_ptr->center) / voxel_size;
        u_ptr[b*3 + 0] = (float)query_local.x();
        u_ptr[b*3 + 1] = (float)query_local.y();
        u_ptr[b*3 + 2] = (float)query_local.z();
        
        valid_b_[b] = has_any;      // need at-least 1 neighbor to have features.
        data->is_valid = false;
    }


    std::vector<float> onnx_output(B*6);        // p2v(3) + sigma + score + phi = 6
    p2v_decoder_onnx->infer(
        feat_host_.data_ptr<float>(), u_host_.data_ptr<float>(), exist_mask_host_.data_ptr<uint8_t>(),
        B, Nn, 
        onnx_output.data());

    vector<float> res_p2v_list(B*3);
    vector<float> res_sigma_list(B);
    
    for (int b = 0; b < B; ++b) {
        const float* out = onnx_output.data() + b*6;
        res_p2v_list[b*3] = out[0];
        res_p2v_list[b*3 + 1] = out[1];
        res_p2v_list[b*3 + 2] = out[2];
        res_sigma_list[b] = out[3];
    }

    int num_valid = 0;
    for (int b = 0; b < B; ++b) {
        if (!valid_b_[b]) 
            continue;
        const float sigma_d_val = res_sigma_list[b];
        const float* p2v_value  = &res_p2v_list[3*b];

        auto *rd = batch_list[b];
        rd->vec = V3D(p2v_value[0], p2v_value[1], p2v_value[2]) * voxel_size;
        rd->sigma_d = sigma_d_val;
        rd->is_valid = true;
        ++num_valid;
    }
    return num_valid;
}




void ImplicitVoxelMap::ensure_cpu_buffers(int B, int Nn) {
    // GPU-CPU data transfer consumes too-much time, and I cannot handle it well.
    // So the current model is running on pure CPU.
    auto host_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto host_u8  = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);

    bool need_alloc = false;
    if (!feat_host_.defined() || feat_host_.sizes() != torch::IntArrayRef{B, Nn, 64})
        need_alloc = true;

    if (need_alloc) {
        feat_host_        = torch::zeros({B, Nn, 64}, host_f32);
        u_host_           = torch::empty({B, 3},     host_f32);
        exist_mask_host_  = torch::zeros({B, Nn},     host_u8);
        valid_b_.assign(B, false);
    } else {
        feat_host_.zero_();
        exist_mask_host_.zero_();
        valid_b_.assign(B, false);
        u_host_.zero_();
    }
}

