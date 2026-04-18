#include "ObsScheduler.hpp"
#include "ObsScheduler.hpp"
#include "common_lib.h"
#include <Eigen/Eigenvalues>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <pcl/common/io.h>
#include <pcl/impl/point_types.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include "my_tools.hpp"
#include "tf/LinearMath/Transform.h"

#include <random>
#include <algorithm>
#include <vector>
#include <cmath>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


static inline float clamp01(float x) {return (x < 0.f) ? 0.f : (x > 1.f ? 1.f : x);}

// Calculate the max eigen-value of a matrix.
inline double calcLambdaMax(const Eigen::Matrix3d& Sigma){
    Eigen::Matrix3d S = 0.5 * (Sigma + Sigma.transpose());
    S += 1e-12 * Eigen::Matrix3d::Identity();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(S);
    if (es.info() != Eigen::Success) {
        return std::max(1e-12, S.trace() / 3.0);
    }
    return std::max(1e-12, (double)es.eigenvalues()(2));
}



void ObsScheduler::catchCurrentScan(const PointCloudXYZI& scan_body, const PointCloudXYZI& scan_world){
    m_scan_.lock();
    pcl::copyPointCloud(scan_body, scan_body_);
    pcl::copyPointCloud(scan_world, scan_world_);
    m_scan_.unlock();
}



/*******************************************************************************************
     * IMPORTANCE SAMPLING BASED ON VOXEL MARGIN DISTANCE
     * 
     * This function implements an importance sampling strategy for point cloud observations.
     * The key insight: points near voxel boundaries (margins) are tend to apper in different voxels.
     * We assign weights based on a probabilistic framework:
     * 
     * 1. For each point, we identify its nearest voxel face (xp/xn/yp/yn/zp/zn) based on 
     *    maximum absolute offset from voxel center.
     * 
     * 2. We compute two distances along the normal direction:
     *    - d_center: distance to the center voxel boundary (0.5 * voxel_size)
     *    - d_support: distance to the extended observation region (1.5 * voxel_size)
     * 
     * 3. Using the point's uncertainty covariance projected onto the normal direction,
     *    we compute a probability weight w = erf(z_center / √2), which represents the
     *    likelihood that the point lies within its current center voxel given the noise.
     *    Points with lower probability (closer to boundaries) receive higher sampling
     *    priority because they provide more good observations.
     * 
     * This sampling has two benefits: 1) only use good observations (not near voxel margins), and 2) less point to run faster
*******************************************************************************************/
void ObsScheduler::rankObservations(
    const std::vector<M3D>* covs_world_ptr,
    const std::vector<bool>* valid_ptr,
    double voxel_size)
{
    obs_items_.resize(0);
    if (!covs_world_ptr || !valid_ptr) 
        return ;
    const auto& covs  = *covs_world_ptr;
    const auto& valid = *valid_ptr;

    const int N = static_cast<int>(scan_world_.points.size());
    if ((int)covs.size() != N || (int)valid.size() != N) 
        return ;

    const double half_center  = 0.5 * voxel_size;   // half of voxel size
    const double half_support = 1.5 * voxel_size;   // half of 3-voxel size (using 3*3*3 neighbor)

    const double eps = 1e-12;
    const double inv_sqrt2 = 0.7071067811865475; // 1/sqrt(2)

    obs_items_.resize(N);

    for (int i = 0; i < N; ++i) {
        if (!valid[i]) {
            obs_items_[i].is_valid = false;
            continue;
        }

        const auto& pw = scan_world_.points[i];
        const auto& pb = scan_body_.points[i];

        // 1) Determine voxel center based solely on coordinates (independent of map)
        const double inv_s = 1.0 / voxel_size;
        const int ix = (int)std::floor(pw.x * inv_s);
        const int iy = (int)std::floor(pw.y * inv_s);
        const int iz = (int)std::floor(pw.z * inv_s);

        const double cx = (ix + 0.5) * voxel_size;
        const double cy = (iy + 0.5) * voxel_size;
        const double cz = (iz + 0.5) * voxel_size;

        // Local coordinates within voxel u \in [-0.5s, 0.5s]
        const double ux = pw.x - cx;
        const double uy = pw.y - cy;
        const double uz = pw.z - cz;

        // 2) Find nearest plane: xp/xn/yp/yn/zp/zn
        // Equivalent to: choose axis with maximum |u|; sign determines +/- face
        const double ax = std::abs(ux);
        const double ay = std::abs(uy);
        const double az = std::abs(uz);

        int axis = 0; // 0:x, 1:y, 2:z
        if (ay >= ax && ay >= az) axis = 1;
        else if (az >= ax && az >= ay) axis = 2;

        const double u_axis = (axis == 0 ? ux : (axis == 1 ? uy : uz));
        const double a_axis = std::abs(u_axis);
        const int sign = (u_axis >= 0.0) ? +1 : -1; // +: p-plane, -: n-plane

        // Normal vector n of nearest plane (only can be ±x/±y/±z)
        Eigen::Vector3d n(0.0, 0.0, 0.0);
        n(axis) = (double)sign;

        // 3) Calculate minimum distances dmin to "center voxel boundary (0.5s)" and "observation region boundary (1.5s)"
        // Since nearest plane is along axis, distance is simply half - |u_axis|
        double d_center  = half_center  - a_axis;   // distance to nearest plane of center voxel

        // Numerical protection (points may slightly exceed voxel range due to errors)
        if (d_center  < 0.0) d_center  = 0.0;

        // 4) Directional variance: n^T Sigma n
        // Since n is an axial unit vector (±), this is simply Sigma(axis,axis) (sign doesn't matter)
        Eigen::Matrix3d Sigma = covs[i];
        Sigma = 0.5 * (Sigma + Sigma.transpose()); // symmetrize
        double var_n = Sigma(axis, axis);
        if (var_n < eps) var_n = eps;

        const double sigma_n = std::sqrt(var_n);

        // 5) If not exceeding observation region, assign weight w using probability of "center voxel (0.5s)"
        // Probability quantity given: dmin / sqrt(n^T Sigma n) = z_center
        // Map to interval probability: w = erf(z_center / sqrt(2)) = P(|e|<=d_center) (1D Gaussian)
        const double z_center = d_center / sigma_n;                 // target probability metric
        double w = std::erf(z_center * inv_sqrt2);                         // map to [0,1)
        if (w < 0.0) w = 0.0;
        if (w > 1.0) w = 1.0;

        ObsItem it;
        it.idx = i;
        it.w = (float)w;
        it.is_valid = true;
        it.pos_world = V3D(pw.x, pw.y, pw.z);
        it.pos_body = V3D(pb.x, pb.y, pb.z);
        it.cov = Sigma;
        it.is_selected = false;     // default: not selected
        obs_items_[i] = it;         // save to data structure
    }

    return ;
}


void ObsScheduler::pubObservations(void){
    if (!pub_sel_obs_){
        cout << "Publisher not initialized. Returning." << endl;
        return;
    }
    if (obs_items_.empty()){
        cout << "obs_items is empty. Returning." << endl;
        return;
    } 
    
    PointCloudXYZI pc_w;
    pc_w.reserve(obs_items_.size());

    const auto& cloud = scan_world_;
    const int N = static_cast<int>(cloud.points.size());


    for (const auto& it : obs_items_) {
        const int idx = it.idx;
        // if (idx < 0 || idx >= N || !it.is_valid) continue;
        if(!it.is_valid) continue;

        const auto& p = cloud.points[idx];

        // 2) weighted_obs: write weight w to intensity field
        PointType q = p;
        q.intensity = it.w;   // weight in [0,1]
        pc_w.points.push_back(q);
    }

    pc_w.width = (uint32_t)pc_w.points.size();
    pc_w.height = 1;
    pc_w.is_dense = false;

    sensor_msgs::PointCloud2 msg_selected;
    pcl::toROSMsg(select_obs_, msg_selected);
    msg_selected.header.stamp = ros::Time::now();
    msg_selected.header.frame_id = "camera_init";
    pub_sel_obs_.publish(msg_selected);     // publish selected points.
}



// Sample observations based on importance. 
void ObsScheduler::samplingByWeight(void){

    select_obs_.clear();
    if (obs_items_.empty()) return;

    // 1) Collect available points
    struct Candidates {
        int item_i;   // index in obs_items_
        float w;
    };
    std::vector<Candidates> cands;
    cands.reserve(obs_items_.size());

    for (int i = 0; i < (int)obs_items_.size(); ++i) {
        auto& it = obs_items_[i];
        if (it.w <= 0.f) continue;
        cands.push_back({i, it.w});
    }

    const int M = (int)cands.size();
    if (M == 0) return;

    // 2) If not enough observations, select all for further observations.
    if (params_.obs_select_K >= M) {
        select_obs_.points.reserve(M);
        for (const auto& c : cands) {
            auto& it = obs_items_[c.item_i];
            PointType p;
            p.x = (float)it.pos_world.x();
            p.y = (float)it.pos_world.y();
            p.z = (float)it.pos_world.z();
            p.intensity = it.w;                 // for visualization
            select_obs_.points.push_back(p);
            it.is_selected = true;
        }
        select_obs_.width  = (uint32_t)select_obs_.points.size();
        select_obs_.height = 1;
        select_obs_.is_dense = false;
        return;
    }

    // 3) K < M: Weighted sampling without replacement (Efraimidis–Spirakis algorithm)
    // key = U^(1/w), select K items with largest keys
    // Larger w → key tends to be larger → higher probability of being selected
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> uni(1e-12, 1.0); // avoid log(0)

    struct Keyed {
        double key;
        int item_i;
    };
    std::vector<Keyed> keyed;
    keyed.reserve(M);

    for (const auto& c : cands) {
        const double w = std::max(1e-6, (double)c.w);
        const double u = uni(gen);
        const double key = std::pow(u, 1.0 / w);
        keyed.push_back({key, c.item_i});
    }

    // Select K items with largest keys: use nth_element O(M)
    std::nth_element(
        keyed.begin(),
        keyed.begin() + params_.obs_select_K,
        keyed.end(),
        [](const Keyed& a, const Keyed& b){ return a.key > b.key; } // descending order
    );

    // First K items are the samples (order doesn't matter; can sort by key if desired)
    select_obs_.points.reserve(params_.obs_select_K);
    for (int i = 0; i < params_.obs_select_K; ++i) {
        auto& it = obs_items_[keyed[i].item_i];
        PointType p;
        p.x = (float)it.pos_world.x();
        p.y = (float)it.pos_world.y();
        p.z = (float)it.pos_world.z();
        p.intensity = it.w;                 // for visualization
        select_obs_.points.push_back(p);
        it.is_selected = true;
    }

    // For all "selected" points, calculate observations.
    for(auto& it: obs_items_){
        it.lambda_max = calcLambdaMax(it.cov);      // used for subsequent covariance modification
    }

    select_obs_.width  = (uint32_t)select_obs_.points.size();
    select_obs_.height = 1;
    select_obs_.is_dense = false;
}


void ObsScheduler::updateState(const state_ikfom& state_in){
    transformLidar(state_in, scan_body_, scan_world_);
}

