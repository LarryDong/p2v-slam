
#include <malloc.h>
#include <chrono>
#include <iomanip>
#include <memory>
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <algorithm>
#include <string>

#include <torch/script.h>

#include <Eigen/Core>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver/CustomMsg.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
// #include <pcl/io/ply_io.h>


#include "common_lib.h"
#include "ros/console.h"
#include "ros/publisher.h"
#include "so3_math.h"

#include "IMU_Processing.hpp"
#include "my_tools.hpp"
#include "preprocess.hpp"
// #include "voxel_map_util.hpp"
#include "viewer.hpp"

#include "PointCloudUn.hpp"
#include "implicit_voxel_map.hpp"
#include "ObsScheduler.hpp"


#define INIT_TIME       (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN        (720000)
#define PUBFRAME_PERIOD     (20)



/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   time_sync_en = false;
double lidar_time_offset = 0.0;

bool b_output_debug_info = false;
bool b_publish_ivoxel_map = false;
bool b_pub_selected_obs = false;
bool b_pub_p2v_matches = false;
int g_cpu_number = 1;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;

mutex mtx_buffer;
condition_variable sig_buffer;
string  lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_surf_min = 0;
double total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false;
bool   scan_pub_en = false, b_dense_pub_en = true;


vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>             time_buffer;
deque<PointCloudXYZI::Ptr>    lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;


PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_full_world(new PointCloudXYZI());     // feats_undistort -> transfer
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());      // feats_undistort -> ds
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());     // feats_down_body -> transfer

PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));

std::vector<M3D> var_down_body;

pcl::VoxelGrid<PointType> downSizeFilterSurf;

std::vector<float> nn_dist_in_feats;
std::vector<float> nn_plane_std;
// PointCloudXYZI::Ptr feats_with_correspondence(new PointCloudXYZI());

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

// params for voxel mapping algorithm
double min_eigen_value = 0.003;
int max_layer = 0;

int max_cov_points_size = 50;
int max_points_size = 50;
double sigma_num = 2.0;
double max_voxel_size = 1.0;

double ranging_cov = 0.0;
double angle_cov = 0.0;
std::vector<double> layer_point_size;

int publish_max_voxel_layer = 0;


/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
// vect3 pos_lid;           //~ not used

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());


////////////////////////////////////////  Dong Yan Global  ////////////////////////////////////////
// Global maps. Voxel map + Implicit map
std::shared_ptr<ImplicitVoxelMap> g_implicit_map_ptr;
// bool flg_implicitMap_inited = false;     // use a different name. 2026.02.24
double implicitMap_init_time = 2.0;           // after 2s to init the map.
std::shared_ptr<MyViewer> g_viewer;
int g_scan_id = 0;
std::ofstream debug_cov_file;

// Uncertainty parts.
std::shared_ptr<PointCloudUn> g_un_pc_ptr;      // undertainty pointcloud ptr


// Scheduler parts
std::shared_ptr<ObsScheduler> g_schduler_ptr;

// Data sharing
std::vector<int> g_selected_idx;
PointCloudXYZI::Ptr g_selected_obs_ptr(new PointCloudXYZI());
std::vector<Eigen::Matrix3d>* g_covs_world_ptr;
std::vector<bool>* g_cov_valid_ptr;


MyTimer g_timer;

////////////////////////////////////////  Dong Yan Global  ////////////////////////////////////////



void SigHandle(int sig){
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po) {
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg){
    auto time_offset = lidar_time_offset;
//    std::printf("lidar offset:%f\n", lidar_time_offset);
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() + time_offset < last_timestamp_lidar){
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec() + time_offset);
    last_timestamp_lidar = msg->header.stamp.toSec() + time_offset;
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}


double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg){
    g_scan_id++;
    cout << "------------------- Get lidar scan number: " << g_scan_id << " ------------------- " << endl;
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar){
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() ){
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty()){
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}


void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in){
    publish_count ++;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en){
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    if (timestamp < last_timestamp_imu){
        ROS_WARN("imu loop back, ignoring!!!");
        ROS_WARN("current T: %f, last T: %f", timestamp, last_timestamp_imu);
        return;
    }
    // delete abnormal  data
    if (std::abs(msg->angular_velocity.x) > 10  || std::abs(msg->angular_velocity.y) > 10 || std::abs(msg->angular_velocity.z) > 10) {
        ROS_WARN("Large IMU measurement!!! Drop Data!!! %.3f  %.3f  %.3f", msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
        return;
    }

    last_timestamp_imu = timestamp;
    mtx_buffer.lock();
    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}


double lidar_mean_scantime = 0.0;
int scan_num = 0;

bool sync_packages(MeasureGroup &meas) {

    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed){
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime){
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else{
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}



void publish_frame_world(const ros::Publisher & pubLaserCloudFull){
    PointCloudXYZI::Ptr laserCloudFullRes(b_dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI laserCloudWorld;
    for (int i = 0; i < size; i++){
        PointType const * const p = &laserCloudFullRes->points[i];
        if(p->intensity < 1){       // ignore low-reflection points.
            continue;
        }
        PointType p_world;
        RGBpointBodyToWorld(p, &p_world);
        laserCloudWorld.push_back(p_world);
    }
    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(laserCloudWorld, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFull.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}


template<typename T>
void set_posestamp(T & out){
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;

}

void publish_odometry(const ros::Publisher & pubOdomAftMapped){
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++){
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform           transform;
    tf::Quaternion          q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, odomAftMapped.pose.pose.position.y, odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );

    static tf::TransformBroadcaster br_world;
    transform.setOrigin(tf::Vector3(0, 0, 0));
    q.setValue(p_imu->Initial_R_wrt_G.x(), p_imu->Initial_R_wrt_G.y(), p_imu->Initial_R_wrt_G.z(), p_imu->Initial_R_wrt_G.w());
    transform.setRotation(q);
    br_world.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "world", "camera_init"));
}


void publish_path(const ros::Publisher pubPath){
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";
    static int jjj = 0;
    jjj++;
    if (jjj % 5 == 0){
        path.header.stamp = msg_body_pose.header.stamp;
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}



// Point-to-voxel observation model
void build_p2v_measurement(const std::vector<ObsItem>& obs_items, std::vector<UnifiedMeasurement>& meas, int iteration){
    if (!g_implicit_map_ptr->is_map_inited)
        return;
    if (obs_items.empty())
        return;

    std::vector<p2v> p2v_list;
    std::vector<double> lambda_list;  // eigen-values of each point uncertainty cov. (for observation selection and modification)
    p2v_list.reserve(obs_items.size());
    lambda_list.reserve(obs_items.size());

    // Prepare for P2V obs.
    for (const auto& it : obs_items){
        if (!it.is_selected) continue;          // now the obs_items are all "selected" items.
        p2v item;
        item.body        = it.pos_body;
        item.query_world = it.pos_world;
        p2v_list.push_back(item);
        lambda_list.push_back((it.lambda_max > 0.0) ? it.lambda_max : 0.0);
    }

    if (p2v_list.size() < 10){
        std::cout << "Not enough selected observations: " << p2v_list.size() << std::endl;
        return;
    }

    // build residuals by network.
    int valid_cnt = g_implicit_map_ptr->buildResidual(p2v_list);

    meas.clear();
    meas.reserve(p2v_list.size());
    
    for (size_t i = 0; i < p2v_list.size(); ++i){
        const p2v& p = p2v_list[i];
        if (!p.is_valid) continue;

        UnifiedMeasurement m;
        m.point = p.body;
        m.world = p.query_world;
        m.norm  = p.vec.normalized();
        m.dist  = p.vec.norm();

        const double sigma = std::max(1e-12, (double)p.sigma_d);
        const double lambda_max = lambda_list[i];
        const double yita = 0.001;                  // an experience value to lower the match quality based on lidar noise model
        const double var_new = sigma + yita * std::max(0.0, lambda_max);  // larger lidar cov, p2v-match has lower weight
        m.R_inv = 1.0 / var_new * 50;               // this calculation is from original PV-LIO, and I did not modify it.
        meas.push_back(m);
    }

    if (meas.empty()){
        std::cout << "Warning: No valid p2v measurements." << std::endl;
    }

    if (b_pub_p2v_matches && iteration == 1) {      // publish p2v-matches for vis.
        g_viewer->setMatches(p2v_list);
        g_viewer->publishPointAndMatch();
    }
    if(b_output_debug_info)
        std::cout << "P2V obs. Iter: " << iteration << ", obs_num: total / valid : " << p2v_list.size() << " / " << valid_cnt << std::endl;
}


/////////////////////////////////////////////////////////////////////////////////////////////
///                       New H model (using ObsItem selected set)
/////////////////////////////////////////////////////////////////////////////////////////////
void p2v_observation_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data, int iteration) {

    // 0. check stop iteration
    double time_used = g_timer.imu_forward_ + g_timer.calc_point_un_ + g_timer.ekf_scheduler_ + g_timer.ekf_p2v_;
    bool stop_flag = g_schduler_ptr->checkStopIteration(iteration, time_used);
    if (stop_flag) {
        ekfom_data.valid = false;
        cout << "Stop oberation." << endl;
        return;
    }

    g_timer.iteration_cnt_ = iteration;
    std::vector<UnifiedMeasurement> measurements;
    double t0_scheduler = omp_get_wtime();

    // 1. Update uncertainty pointcloud registration using new state.
    g_un_pc_ptr->updateState(s);
    g_covs_world_ptr = &g_un_pc_ptr->v_covs_register_;

    // 2. Update schduler observation selection. (Select "good" observations)
    g_schduler_ptr->updateState(s);                                      // update scan_world_, to check importance (boundary check)
    g_schduler_ptr->rankObservations(g_covs_world_ptr, g_cov_valid_ptr); // calculate each point's importance (based on cov. Larger cov, less important)
    g_schduler_ptr->samplingByWeight();                                  // importance sampling

    if(b_pub_selected_obs && iteration == 1){       // publish the selected point for p2v-obervation build.
        g_schduler_ptr->pubObservations();
    }

    g_timer.ekf_scheduler_ += omp_get_wtime() - t0_scheduler; // just keep the latest. will be reset every time.
    double t0_p2v = omp_get_wtime();
    if (g_implicit_map_ptr->is_map_inited) {
        build_p2v_measurement(g_schduler_ptr->getSelObsItems(), measurements, iteration);
    }
    g_timer.ekf_p2v_ += omp_get_wtime() - t0_p2v;
    g_timer.obs_number_ = measurements.size();

    if (measurements.empty()) {
        ekfom_data.valid = false;
        ROS_WARN_COND(g_implicit_map_ptr->is_map_inited, "No Effective points!");
        return;
    }

    // cout << "Observation model. obs_num: " << measurements.size() << endl;
    ekfom_data.valid = true;
    effct_feat_num = (int)measurements.size();
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);
    ekfom_data.h.resize(effct_feat_num);
    ekfom_data.R.resize(effct_feat_num, 1);

    for (int i = 0; i < effct_feat_num; i++) {
        const UnifiedMeasurement &m = measurements[i];
        V3D point_this_be(m.point);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_this);
        V3D norm_vec(m.norm);
        V3D C(s.rot.conjugate() * norm_vec);
        V3D A(point_crossmat * C);
        ekfom_data.h_x.block<1, 12>(i, 0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        ekfom_data.h(i) = m.dist;
        ekfom_data.R(i) = m.R_inv;
    }
}


// Feature extraction loop
int g_cnt_update_acc = 0;
ros::Time g_last_print_time;
void voxelFeatureExtraction(void){

    // cout <<"voxelFeatureExtraction loop. " << endl;
    const int Nn = 27;
    const int K = 40;
    const int C = 3;    
    torch::Device cpu = torch::kCPU;
    torch::Device device = g_implicit_map_ptr->device;
    auto opt_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(cpu);
    auto opt_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(cpu);

    int cnt_update = 0;
    implicitUnorderMap& ivmap = g_implicit_map_ptr->implicit_map;
    // 1. Copy temp points to voxels
    static int dbg_cnt_needupdate = 0;
    for(auto& item : ivmap){
        std::shared_ptr<ImplicitVoxelGrid> vg = item.second;
        if(!vg)
            ROS_ERROR("Null voxel grid. Should not happen.");
        
        if(vg->is_full && vg->has_feature)      // has feature and up-to 50 points, skip.
            continue;

        bool flag_need_update = false;
        vg->m_pointadd.lock();
        // check if need to add points.
        if(vg->tmp_points.size() > vg->update_thresh) {     // need update points.
            int num_to_add = vg->tmp_points.size();

            // move tmp points to formal.
            for(const V3D& p : vg->tmp_points){
                vg->points.push_back(p);
                vg->norm_points.push_back((p - vg->center) / vg->voxel_size);
            }
            vg->tmp_points.resize(0);
            if(vg->norm_points.size() > vg->predictable_thresh){
                flag_need_update = true;
                dbg_cnt_needupdate++;
            }
        }
        vg->m_pointadd.unlock();

        // extract features.
        if(flag_need_update){
            torch::Tensor pts_v = torch::zeros({1, K, C}, opt_f32);     // [1, K, 3]
            torch::Tensor mask_v = torch::zeros({1, K}, opt_u8);        // [1, K] point mask
            float *pts_ptr = pts_v.data_ptr<float>();                   // this process  should be on CPU. So convert to GPU later.
            uint8_t *mask_ptr = mask_v.data_ptr<uint8_t>();
    
            const auto& pts = vg->norm_points;
            const int copyN = std::min<int>(pts.size(), K);     // number to sampe in each voxel      
            int seed = 42;
            std::vector<int> indices = my_tools::select_k_indices(pts, K, seed);
            for (int j = 0; j < indices.size(); ++j) {
                int idx = indices[j];
                int base = j * C;
                pts_ptr[base + 0] = pts[idx].x();
                pts_ptr[base + 1] = pts[idx].y();
                pts_ptr[base + 2] = pts[idx].z();
                mask_ptr[j] = 1;
            }

            // New onnx inference
            std::vector<float> feat_vec = g_implicit_map_ptr->p2v_encoder_onnx->infer(pts_ptr,mask_ptr,K);
            // back-to torch Tensor
            torch::Tensor feat_onnx = torch::from_blob(feat_vec.data(),{64},torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();

            vg->m_feat.lock();              // conflict for p2v decoder.
            vg->feature = feat_onnx;
            vg->has_feature = true;         // always true if the first feature is get.
            vg->m_feat.unlock();
            cnt_update++;
        }
    }

    int dbg_cnt_hasfeat = 0;
    for(auto& item : ivmap){
        if(item.second->has_feature)
            dbg_cnt_hasfeat++;
    }

    g_cnt_update_acc += cnt_update;
    auto now = ros::Time::now();
    if ((now - g_last_print_time).toSec() >= 1.0) {
        if(b_output_debug_info)
            cout << "[Voxel Update Thread] Updated: " << g_cnt_update_acc << " voxels in last 1s" << endl;
        g_cnt_update_acc = 0;
        g_last_print_time = now;
        dbg_cnt_needupdate = 0;
    }
    return ;
}


void voxelFeatureExtractionLoop(void){
    cout << "voxelFeatureExtractionLoop start to get features 10Hz." << endl;
    g_last_print_time = ros::Time::now();
    while(ros::ok()){
        if(g_implicit_map_ptr->is_map_inited){
            voxelFeatureExtraction();       // extract feature every 100ms

            {   // map memory control
                if(g_scan_id %100 == 0)            // every 10s, clean the RSS memory.
                    malloc_trim(0);
            }
        }
        else
            // cout << "Waiting for implicitMap init." << endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));        // 10Hz loop
    }
}



/////////////////////////////////////////////////////////////////////////////////////////////
///                                             MAIN
/////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    // model
    int batch_size_yaml;
    string encoder_yaml, decoder_yaml;
    {   // State estimation control
        int obs_select_K = 0;
        double iteration_time_budget = 0.0;
        nh.param<int>("estimation/select_obs", obs_select_K, 500);
        nh.param<double>("estimation/iteration_time_budget", iteration_time_budget, 100);
        nh.param<int>("estimation/max_iteration", NUM_MAX_ITERATIONS, 4);
        nh.param<vector<double>>("estimation/extrinsic_T", extrinT, vector<double>());
        nh.param<vector<double>>("estimation/extrinsic_R", extrinR, vector<double>());
        g_schduler_ptr = std::make_shared<ObsScheduler>();
        g_schduler_ptr->init(nh);
        g_schduler_ptr->setParams(obs_select_K, iteration_time_budget);
    }


    {
        // Topic control
        nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
        nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
        nh.param<bool>("common/time_sync_en", time_sync_en, false);

        // preprocess
        nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
        nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
        nh.param<float>("preprocess/det_range",DET_RANGE,300.f);
        nh.param<double>("preprocess/down_sample_size", filter_size_surf_min, 0.5);
        nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 1);
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

        // noise model params
        nh.param<double>("noise_model/ranging_cov", ranging_cov, 0.02);
        nh.param<double>("noise_model/angle_cov", angle_cov, 0.05);
        nh.param<double>("noise_model/gyr_cov",gyr_cov,0.1);
        nh.param<double>("noise_model/acc_cov",acc_cov,0.1);
        nh.param<double>("noise_model/b_gyr_cov",b_gyr_cov,0.0001);
        nh.param<double>("noise_model/b_acc_cov",b_acc_cov,0.0001);

        g_un_pc_ptr = std::make_shared<PointCloudUn>();
        g_un_pc_ptr->init(nh);
        g_un_pc_ptr->setUncertConfig(ranging_cov, angle_cov);


        // Point-to-Voxel Network and iVoxMap Control
        nh.param<int>("p2v/model/batch_size", batch_size_yaml, 1);
        nh.param<string>("p2v/model/encoder", encoder_yaml, "default.encoder.pt");
        nh.param<string>("p2v/model/decoder", decoder_yaml, "default.decoder.pt");
        nh.param<int>("p2v/num_cpu", g_cpu_number, 4);          // how many CPU core do you have.

        // Publish & Rviz control
        nh.param<bool>("publish/pub_dense_registered_scan",b_dense_pub_en, true);
        nh.param<bool>("publish/pub_ivoxel_map", b_publish_ivoxel_map, false);
        nh.param<int>("publish/publish_max_voxel_layer", publish_max_voxel_layer, 0);
        nh.param<bool>("publish/pub_selected_obs", b_pub_selected_obs, 0);
        nh.param<bool>("publish/pub_p2v_matches", b_pub_p2v_matches, 0);


        // Debug Info Control
        nh.param<bool>("output_debug_info", b_output_debug_info, true);
    }


    cout << "---------------------------------------------" << endl;
    cout << "configs from yaml: " << endl;
    cout << "LiDAR type: " << p_pre->lidar_type << endl;
    cout << "Batch size: " << batch_size_yaml << endl;
    cout << "Encoder (VE-Net) model: " << encoder_yaml << endl;
    cout << "Decoder (IR-Net) model: " << decoder_yaml << endl;
    cout << "CPU Core Number: " << g_cpu_number << endl;
    cout << "---------------------------------------------" << endl;

    // some initialization.
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";
    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, p2v_observation_model, NUM_MAX_ITERATIONS, epsi);

    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/Odometry", 100000);
    ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 100000);

    /////////////////////////////////////// My new codes ///////////////////////////////////////
    g_viewer = std::make_shared<MyViewer>(nh);
    g_viewer->init();

    // iVoxMap settings.
    const int max_point_thresh = 100;
    const int predictable_thresh = 5;
    const int update_thresh = 5;
    const double voxel_size = 0.3;
    torch::Device dev(torch::kCPU);
    g_implicit_map_ptr = std::make_shared<ImplicitVoxelMap>(&nh, max_point_thresh, predictable_thresh, update_thresh, voxel_size, encoder_yaml, decoder_yaml, dev, batch_size_yaml);

    PointCloudXYZI wait_to_init_pc;                 // accumulated pointcloud for iVoxMap initialization
    std::thread t(voxelFeatureExtractionLoop);      // start a thread to extract features at 10Hz. ATTEN: after ptr inited.

    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    int scan_idx = -1;

    ////////////////////////////////////////  Main Loop  ////////////////////////////////////////
    while (ros::ok()){
        if (flg_exit) break;
        ros::spinOnce();

        if(sync_packages(Measures)){
            if (flg_first_scan){
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            scan_idx++;
            g_timer.reset();
            g_timer.idx_ = scan_idx;
            
            double t_total = omp_get_wtime();
            double t_forward = omp_get_wtime();


            // I. Forward prediction & undistortion
            {
                p_imu->Process(Measures, kf, feats_undistort);
                state_point = kf.get_x();   // every time forward, get state_point.
                g_timer.imu_forward_ = omp_get_wtime() - t_forward;
                downSizeFilterSurf.setInputCloud(feats_undistort);
                downSizeFilterSurf.filter(  *feats_down_body);
                my_tools::removeInvalidPoint(*feats_down_body, p_pre->blind, DET_RANGE);
                transformLidar(state_point, *feats_down_body, *feats_down_world);
            }

            if (feats_down_body->empty() || (feats_down_body == NULL)) {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            {   // iVoxMap init. After 2s to accumulate and init.
                if(g_implicit_map_ptr->is_map_inited == false){
                    // first accumulate pointclouds.
                    transformLidar(state_point, *feats_undistort, *feats_full_world);
                    wait_to_init_pc += *feats_full_world;
                    cout << "... wait_to_init_pc size: " << wait_to_init_pc.points.size() << endl;
                    my_tools::dsPointCloud(wait_to_init_pc, 0.05);

                    // if accumulate enough time, just init.
                    bool enough_time_flag = (Measures.lidar_beg_time - first_lidar_time) < implicitMap_init_time ? false : true;
                    if(enough_time_flag){
                        vector<V3D> pts = my_tools::pcl2vec(wait_to_init_pc.makeShared());  // using accumulated points as input
                        g_implicit_map_ptr->build(pts);
                        voxelFeatureExtraction();   // extract all features first and then set to ture.
                        g_implicit_map_ptr->is_map_inited = true;
                        cout << "<-- Finish building implicit_map." << endl;
                        if(b_output_debug_info){
                            cout << "Points used for build: " << wait_to_init_pc.points.size() << endl;
                            g_implicit_map_ptr->printMapInfo();
                        }
                    }
                }
            }


            // II. Point Cloud Uncertainty calculation. Used for p2v observation selection and modification.
            {
                double t0_un = omp_get_wtime();
                g_un_pc_ptr->catchCurrentScan(*feats_down_body, *feats_down_world);
                g_un_pc_ptr->catchCurrentState(state_point);
                g_un_pc_ptr->calcPointCov();
                g_covs_world_ptr = &g_un_pc_ptr->v_covs_register_;
                g_cov_valid_ptr = &g_un_pc_ptr->v_cov_valid_;
                g_timer.calc_point_un_ = omp_get_wtime() - t0_un;
            }
            


            // III. Observation Scheduler process. Select "good" observations based on uncertainty.
            {
                // Just save feats_down_body, world pointcloud will be re-calculated every time the state changes in the iteration.
                g_schduler_ptr->catchCurrentScan(*feats_down_body, *feats_down_world);      
            }


            // IV. IESKF
            {
                double t0_update = omp_get_wtime();
                kf.update_iterated_dyn_share_diagonal();        // `p2v_observation_model`
                state_point = kf.get_x();                       // Attention! Always update the state points.
                double te_update = omp_get_wtime();
                g_timer.ekf_total_ = te_update - t0_update;
            }


            // IV. Map update.
            {
                double tmap= omp_get_wtime();
                transformLidar(state_point, *feats_undistort, *feats_full_world);  // map update use full points.
                vector<V3D> p_list;
                p_list.reserve(feats_full_world->points.size());
                for (size_t i = 0; i < feats_full_world->size(); i++) {
                    V3D point_world(feats_full_world->points[i].x, feats_full_world->points[i].y, feats_full_world->points[i].z);
                    p_list.push_back(point_world);
                }
                g_implicit_map_ptr->update(p_list);
                g_timer.map_update_ = omp_get_wtime() - tmap;
            }

            g_timer.total_ = omp_get_wtime() - t_total;


            // Publish and Output Control.
            if(b_publish_ivoxel_map && scan_idx % 10 == 0){
                g_implicit_map_ptr->publishFullMap();        // if you want to show full map (pointcloud)
                g_implicit_map_ptr->publishVoxelCube();      // if you want to show full implicit map (iVoxMap). REALLY SLOW if your rviz sub it!
            }

            geoQuat.x = state_point.rot.coeffs()[0];   // Used for odometry output
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            publish_odometry(pubOdomAftMapped);         // necessary for trajectory saver.
            publish_frame_world(pubLaserCloudFull);
            publish_path(pubPath);

            g_timer.print();
        }
        rate.sleep();
    }

    return 0;
}
