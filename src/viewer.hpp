#pragma once
#include <pcl/impl/point_types.hpp>
#include <pcl/point_cloud.h>
#include <vector>
#include <Eigen/Eigen>
#include <ros/ros.h>

#include "ros/publisher.h"
#include "voxel_map_util.hpp"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"

using namespace std;

class MyViewer{
    
public:
    MyViewer(const ros::NodeHandle& node) : node_(node){}
    void setMatches(const vector<p2v>& p2v);
    void publishPointAndMatch(void);
    void init(void);
    void reset(void);

public:

    pcl::PointCloud<pcl::PointXYZINormal> p2v_pc_;      // publish point-to-voxel used pointcloud.
    ros::Publisher pub_p2v_pc;

    vector<V3D> p2v_;
    vector<double> p2v_sigma_d_;
    ros::Publisher pub_p2v_marker;    
    ros::NodeHandle node_;
};

extern std::shared_ptr<MyViewer> g_viewer;


