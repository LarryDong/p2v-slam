
#include "viewer.hpp"
#include "ros/duration.h"
#include "ros/time.h"
#include "sensor_msgs/PointCloud2.h"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
#include "my_tools.hpp"
#include <cmath>


void MyViewer::init(void){
    cout <<"[MyViewer::init] init p2v subscriber, and marker publisher" << endl;
    pub_p2v_pc = node_.advertise<sensor_msgs::PointCloud2>("/viewer/full_p2v_pc", 100);
    pub_p2v_marker = node_.advertise<visualization_msgs::MarkerArray>("/viewer/p2v_marker", 100);
}

void MyViewer::reset(void){
    p2v_pc_.points.clear();
    p2v_.clear();
    p2v_sigma_d_.clear();
}

void MyViewer::setMatches(const vector<p2v> &p2v_list){
    reset();
    p2v_pc_.points.reserve(p2v_list.size());
    p2v_.reserve(p2v_list.size());
    p2v_sigma_d_.reserve(p2v_list.size());

    for (const p2v &p : p2v_list) {
        if(p.is_valid){     // only save valid matches.
            p2v_pc_.points.push_back(my_tools::vec2point(p.query_world));
            p2v_.push_back(p.vec);
            p2v_sigma_d_.push_back(p.sigma_d);
        }
    }

}



visualization_msgs::Marker createMarker(int id, int type, const std::string& frame_id) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = ros::Time::now();
    marker.id = id;
    marker.type = type;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.lifetime = ros::Duration(0);
    return marker;
}



void MyViewer::publishPointAndMatch(void){

    // cout << "------------------------ publishPointAndMatch ------------------------" << endl;
    // 1. Publish p2p and p2v pointcloud
    pub_p2v_pc.publish(my_tools::pcl2msg(p2v_pc_));

    // 2. Publish the p2p matches and p2v matches.
    // clear markers
    int id_idx = 0;
    visualization_msgs::MarkerArray clear_markers;
    visualization_msgs::Marker clear_marker;
    clear_marker.id = 0;
    clear_marker.action = visualization_msgs::Marker::DELETEALL;
    clear_marker.header.frame_id = "camera_init";
    clear_markers.markers.push_back(clear_marker);
    pub_p2v_marker.publish(clear_markers);

    // p2v marker with different color
    visualization_msgs::MarkerArray marker_array2;
    visualization_msgs::Marker p2v_lines = createMarker(id_idx++, visualization_msgs::Marker::LINE_LIST, "camera_init");

    const int skip_cnt = 1;         // skip some matches to make speed faster.
    for (int i = 0; i < p2v_.size(); ++i) {
        if (i % skip_cnt != 0)
            continue;

        V3D begin_point = my_tools::point2V3D(p2v_pc_.points[i]);
        V3D end_point = begin_point + p2v_[i];

        geometry_msgs::Point msg_scan_point = my_tools::V3D2GeomsgPoint(begin_point);
        geometry_msgs::Point msg_end_point = my_tools::V3D2GeomsgPoint(end_point);

        visualization_msgs::Marker line_marker = createMarker(id_idx++, visualization_msgs::Marker::LINE_LIST, "camera_init");
        const double match_line_width = 0.05;       // if too thin, the line not clear to be seen.
        line_marker.scale.x = match_line_width;

        // double weight = 1 - std::min(1.0, std::max(0.0, (p2v_sigma_d_[i] - 0.01) / 0.09));       // color by weight.
        // line_marker.color.r = weight;
        // line_marker.color.g = 1-weight;
        line_marker.color.r = 1;
        line_marker.color.g = 0;
        line_marker.color.b = 0;
        line_marker.color.a = 1;

        line_marker.points.push_back(msg_scan_point);
        line_marker.points.push_back(msg_end_point);
        marker_array2.markers.push_back(line_marker);
    }
    pub_p2v_marker.publish(marker_array2);
}



