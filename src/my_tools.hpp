
#ifndef __MY_TOOL_H
#define __MY_TOOL_H

#include <Eigen/Core>
#include <fstream>
#include <iomanip>
#include <pcl/impl/point_types.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <iostream>
#include <vector>

#include <algorithm>
#include <random>
#include <limits>
#include <random>

#include "common_lib.h"
#include "ros/time.h"
#include "sensor_msgs/PointCloud2.h"
#include "use-ikfom.hpp"

#include <omp.h>


inline void transformLidar(const state_ikfom &state_point, const PointCloudXYZI &input_cloud, PointCloudXYZI &trans_cloud){
    trans_cloud.clear();
    for (size_t i = 0; i < input_cloud.size(); i++) {
        pcl::PointXYZINormal p_c = input_cloud.points[i];
        Eigen::Vector3d p_lidar(p_c.x, p_c.y, p_c.z);
        // HACK we need to specify p_body as a V3D type!!!
        V3D p_body = state_point.rot * (state_point.offset_R_L_I * p_lidar + state_point.offset_T_L_I) + state_point.pos;
        PointType pi;
        pi.x = p_body(0);
        pi.y = p_body(1);
        pi.z = p_body(2);
        pi.intensity = p_c.intensity;
        trans_cloud.points.push_back(pi);
    }
}



namespace my_tools{

// 将 PointCloudXYZI::Ptr 转换为 std::vector<Eigen::Vector3d>
inline std::vector<Eigen::Vector3d> pcl2vec(const PointCloudXYZI::Ptr cloud) {
    std::vector<Eigen::Vector3d> points;
    points.reserve(cloud->size());  // 预分配内存提高效率
    
    for (const auto& point : cloud->points) {
        points.emplace_back(point.x, point.y, point.z);
    }
    return points;
}

inline V3D point2V3D(const pcl::PointXYZINormal &p) { return V3D(p.x, p.y, p.z); }
inline pcl::PointXYZINormal V3D2point(const V3D &v) { return pcl::PointXYZINormal(v[0], v[1], v[2]); }
inline geometry_msgs::Point V3D2GeomsgPoint(const Eigen::Vector3d &v3d){
    geometry_msgs::Point mp;
    mp.x = v3d[0];
    mp.y = v3d[1];
    mp.z = v3d[2];
    return mp;
}


inline void dsPointCloud(PointCloudXYZI &cloud, double leaf_size=0.02) {
    PointCloudXYZI::Ptr cloud_in(new PointCloudXYZI), cloud_out(new PointCloudXYZI);
    *cloud_in = cloud;
    pcl::VoxelGrid<pcl::PointXYZINormal> voxel_grid;
    voxel_grid.setInputCloud(cloud_in);
    voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_grid.filter(*cloud_out);
    cloud = *cloud_out;
}



inline sensor_msgs::PointCloud2 pcl2msg(const PointCloudXYZI& cloud){
    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "camera_init";
    return msg;
}


inline void removeInvalidPoint(PointCloudXYZI& cloud, double dis_near=0.01, double dis_far=200) {
    PointCloudXYZI filtered_cloud;
    for (const auto& point : cloud) {
        float distance_sq = point.x * point.x + point.y * point.y + point.z * point.z;
        if (distance_sq >= dis_near*dis_near && distance_sq <= dis_far*dis_far) {
            filtered_cloud.push_back(point);
        }
    }
    cloud = filtered_cloud;
}

inline pcl::PointXYZINormal vec2point(const V3D& v){
    pcl::PointXYZINormal p;
    p.x = v[0];
    p.y = v[1];
    p.z = v[2];
    return p;
}



static inline std::vector<int> fps_indices(const std::vector<Eigen::Vector3d>& pts,int K) {
    const int N = static_cast<int>(pts.size());
    std::vector<int> out;
    if (N == 0 || K <= 0) return out;
    K = std::min(K, N);
    out.reserve(K);
    // --- deterministic init: farthest from centroid ---
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (const auto& p : pts) centroid += p;
    centroid /= static_cast<double>(N);
    int first_idx = 0;
    double best = -1.0;
    for (int i = 0; i < N; ++i) {
        double d = (pts[i] - centroid).squaredNorm();
        if (d > best) {
            best = d;
            first_idx = i;
        }
    }
    std::vector<double> min_dist(N, std::numeric_limits<double>::max());
    std::vector<uint8_t> selected(N, 0);
    out.push_back(first_idx);
    selected[first_idx] = 1;
    for (int i = 0; i < N; ++i) {
        min_dist[i] = (pts[i] - pts[first_idx]).squaredNorm();
    }
    for (int it = 1; it < K; ++it) {
        int best_idx = -1;
        double best_d = -1.0;
        for (int i = 0; i < N; ++i) {
            if (!selected[i] && min_dist[i] > best_d) {
                best_d = min_dist[i];
                best_idx = i;
            }
        }
        if (best_idx < 0) break;
        selected[best_idx] = 1;
        out.push_back(best_idx);
        for (int i = 0; i < N; ++i) {
            if (!selected[i]) {
                double d = (pts[i] - pts[best_idx]).squaredNorm();
                if (d < min_dist[i]) min_dist[i] = d;
            }
        }
    }
    return out;
}


static inline std::vector<int> select_k_indices(const std::vector<Eigen::Vector3d>& pts, int K, uint32_t seed = 42){
    const int N = static_cast<int>(pts.size());
    std::vector<int> idx;
    if (N == 0 || K <= 0) return idx;
    const int copyN = std::min(K, N);
    idx.reserve(copyN);
    return fps_indices(pts, copyN);     // using farthest-point-sampling
}



}

#endif
