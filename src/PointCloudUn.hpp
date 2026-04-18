

#ifndef __POINTCLOUD_UNCERTAINTY_H
#define __POINTCLOUD_UNCERTAINTY_H

#include "ros/node_handle.h"

#include <ros/ros.h>
#include <std_msgs/Int32.h>     // for id callback
#include <sensor_msgs/Imu.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Float32MultiArray.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "common_lib.h"
#include "ros/publisher.h"
#include "use-ikfom.hpp"



class PointCloudUn{
public:
    PointCloudUn(void){;}
    inline void init(ros::NodeHandle &nh) { ; }

    inline void setUncertConfig(double depth_err, double beam_err){
        depth_err_ = depth_err;
        beam_err_ = beam_err;
    }

    void catchCurrentState(const state_ikfom& state_in);
    void updateState(const state_ikfom& state_in);
    void catchCurrentScan(const PointCloudXYZI& scan_body, const PointCloudXYZI& scan_world);
    void calcPointCov(void);

public:
    ros::Publisher pub_uncertainty_, pub_uncertainty_register_;  // publish the each point's uncertainty.
    std::vector<Eigen::Matrix3d> v_covs_, v_covs_register_;     // covarance of each points, and registered covs.
    std::vector<bool> v_cov_valid_;                             // is covarance valid?

private:
    visualization_msgs::Marker createMarker(int id, int type, const std::string& frame_id = "camera_init");
    void calcMeasurementCov(PointType &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov);

    PointCloudXYZI scan_body_, scan_world_;                     // current scan and registered. Using default PonitCloud type.
    Eigen::Vector3d state_pos_;     // not necessary
    Eigen::Matrix3d state_rot_;     // not necessary
    Eigen::Vector3d state_vel_;     // used for position uncertainty estimation
    
    double depth_err_, beam_err_;

    std::mutex m_imu_;
    std::mutex m_lidar_;
    std::mutex m_state_;
};


#endif
