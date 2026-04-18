
#ifndef __OBS_SCHEDULER_H
#define __OBS_SCHEDULER_H


#include <Eigen/Core>
#include <Eigen/Dense>

#include <mutex>
#include <vector>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "common_lib.h"
#include "ros/publisher.h"
#include "sensor_msgs/PointCloud2.h"
#include "use-ikfom.hpp"



struct ObsItem {
    bool is_valid = false;
    int idx = -1;          // index in feats_down_body
    float w = 1.0f;        // weight [0,1]
    V3D pos_world = {0.f, 0.f, 0.f};
    V3D pos_body  = {0.f, 0.f, 0.f};
    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    bool is_selected = false;   // used for observation or not.
    double lambda_max = 0.0f;   // max eigenvalue of covariance matrix
};



class MyTimer{
public:
    MyTimer() { ; }

    // inline double gettime() { return omp_get_wtime(); }
    inline void reset(void){
        idx_ = 0;
        ekf_total_ = ekf_scheduler_ = ekf_p2v_ = 0.0;
        imu_forward_ = calc_point_un_ = map_update_ = total_ = 0.0;
        iteration_cnt_ = 0;
        obs_number_ = 0;
    }

    int idx_;
    double imu_forward_;
    double calc_point_un_;
    double ekf_total_, ekf_scheduler_, ekf_p2v_;    // total = scheduler + p2v + update
    double map_update_;
    double total_;

    int iteration_cnt_;
    int obs_number_;

    inline void print(void){
        cout << "[Timer] idx: " << idx_<< setprecision(2) 
            << ", total: " << total_ *1e3
            << "ms, forward: " << imu_forward_ *1e3
            << "ms, undistort " << calc_point_un_ *1e3
            << "ms, p2v_update: " << ekf_total_*1e3
            << "ms, map_update: " << map_update_ *1e3 
            << "ms " << endl;
    }
};



// ============================
// Scheduler parameters
// ============================
struct ObsSchedulerParams {
    double time_budget_ms = 50.0;
    int obs_select_K = 500;
};



class ObsScheduler {
public:
    ObsScheduler() = default;
    void init(ros::NodeHandle& nh) {
        pub_sel_obs_ = nh.advertise<sensor_msgs::PointCloud2>("/ObsScheduler/obs_selected", 10);
        // v_timer_.reserve(10000);
    }
    inline void setParams(int obs_select_K, double time_budgets) { 
        params_.time_budget_ms = time_budgets; 
        params_.obs_select_K = obs_select_K;
    }

    // If the next iteration time exceeds the time budget, stop the iteration.
    inline bool checkStopIteration(int iteration, double current_dur_s){
        double current_dur_ms = current_dur_s * 1e3;
        double time_per_iter = current_dur_ms / iteration; 
        if(current_dur_ms + time_per_iter > params_.time_budget_ms){
            cout <<"Out iteration time budget: " << current_dur_ms << " + " << time_per_iter << " > " << params_.time_budget_ms << endl;
            return true;
        }
        else
            return  false;
    }

    void catchCurrentScan(const PointCloudXYZI& scan_body, const PointCloudXYZI& scan_world);

    // update only scan_world_, for further check if boundary
    void updateState(const state_ikfom& state_in);

    // get each obsrvs importance
    void rankObservations(const std::vector<M3D>* covs_world_ptr, const std::vector<bool>* valid_ptr, double voxel_size = 0.3);
    void pubObservations(void);
    void samplingByWeight(void);

    inline vector<ObsItem> getSelObsItems(void) const {
        vector<ObsItem> selected_items;
        selected_items.reserve(params_.obs_select_K);
        for (const auto& it : obs_items_) {
            if (it.is_selected) {
                selected_items.push_back(it);
            }
        }
        return selected_items;
    }


private:
    ObsSchedulerParams params_;
    vector<ObsItem> obs_items_;
    std::mutex m_scan_;
    PointCloudXYZI scan_world_, scan_body_;     // input scan
    PointCloudXYZI select_obs_;                 // observations for further process
    ros::Publisher pub_sel_obs_;                // publish `select_obs_` pointcloud.
    int obs_select_num_;
    MyTimer timer_;
};


#endif

