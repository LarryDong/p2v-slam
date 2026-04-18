
#ifndef _IMU_PROCESSING_H
#define _IMU_PROCESSING_H


#include "use-ikfom.hpp"
#include <Eigen/Eigen>
#include <cmath>
#include <common_lib.h>
#include <csignal>
#include <deque>
#include <eigen_conversions/eigen_msg.h>
#include <fstream>
#include <geometry_msgs/Vector3.h>
#include <math.h>
#include <nav_msgs/Odometry.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <so3_math.h>
#include <tf/transform_broadcaster.h>
#include <condition_variable>



/// *************Preconfiguration

#define MAX_INI_COUNT (20)

inline const bool time_list(PointType &x, PointType &y) { return (x.curvature < y.curvature); };

/// *************IMU Process and undistortion
class ImuProcess {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ImuProcess();
        ~ImuProcess();

        void Reset();
        void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
        void set_extrinsic(const V3D &transl, const M3D &rot);
        void set_extrinsic(const V3D &transl);
        void set_extrinsic(const MD(4, 4) & T);
        void set_gyr_cov(const V3D &scaler);
        void set_acc_cov(const V3D &scaler);
        void set_gyr_bias_cov(const V3D &b_g);
        void set_acc_bias_cov(const V3D &b_a);
        Eigen::Matrix<double, 12, 12> Q;
        void Process(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);

        ofstream fout_imu;
        V3D cov_acc;
        V3D cov_gyr;
        V3D cov_acc_scale;
        V3D cov_gyr_scale;
        V3D cov_bias_gyr;
        V3D cov_bias_acc;
        double first_lidar_time;

        SO3 Initial_R_wrt_G;

    private:
        void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
        void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);

        PointCloudXYZI::Ptr cur_pcl_un_;
        sensor_msgs::ImuConstPtr last_imu_;
        deque<sensor_msgs::ImuConstPtr> v_imu_;
        vector<Pose6D> IMUpose;
        vector<M3D> v_rot_pcl_;
        M3D Lidar_R_wrt_IMU;
        V3D Lidar_T_wrt_IMU;
        V3D mean_acc;
        V3D mean_gyr;
        V3D angvel_last;
        V3D acc_s_last;
        double start_timestamp_;
        double last_lidar_end_time_;
        int init_iter_num = 1;
        bool b_first_frame_ = true;
        bool imu_need_init_ = true;
};

#endif

