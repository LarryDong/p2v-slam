
#include "PointCloudUn.hpp"
#include <pcl/common/io.h>


void PointCloudUn::catchCurrentScan(const PointCloudXYZI& scan_body, const PointCloudXYZI& scan_world){
    m_lidar_.lock();
    pcl::copyPointCloud(scan_body, scan_body_);
    pcl::copyPointCloud(scan_world, scan_world_);
    m_lidar_.unlock();
}

void PointCloudUn::updateState(const state_ikfom& state_in){
    m_state_.lock();
    state_rot_ = state_in.rot.toRotationMatrix();
    state_pos_ = state_in.pos;
    state_vel_ = state_in.vel;
    m_state_.unlock();
    // update cov_world
    for(int i=0; i<v_covs_.size(); ++i){
        v_covs_register_[i] = state_rot_ * v_covs_[i] * state_rot_.transpose();   // TODO: issue, need extrinsics. But not used.
    }
}


void PointCloudUn::catchCurrentState(const state_ikfom& state_in){
    m_state_.lock();
    state_rot_ = state_in.rot.toRotationMatrix();
    state_pos_ = state_in.pos;
    state_vel_ = state_in.vel;
    m_state_.unlock();
}


// calculate point uncertainty by "angle+radius" error model.
void PointCloudUn::calcPointCov(void){
    std::vector<Eigen::Matrix3d>().swap(v_covs_);
    std::vector<Eigen::Matrix3d>().swap(v_covs_register_);
    v_cov_valid_.resize(0);

    Eigen::Matrix3d Sigma_measure = Eigen::Matrix3d::Zero();

    for(auto pb : scan_body_.points){
        Eigen::Matrix3d Sigma_total = Eigen::Matrix3d::Zero();
        double x = pb.x;
        double y = pb.y;
        double z = pb.z;
        double r2 = x*x+y*y+z*z;
        if(r2 < 1e-4 || std::isnan(x)){     // 1cm 
            v_covs_.push_back(Eigen::Matrix3d::Zero());
            v_covs_register_.push_back(Eigen::Matrix3d::Zero());
            v_cov_valid_.push_back(false);
            continue;
        }
        calcMeasurementCov(pb, depth_err_, beam_err_, Sigma_measure);
        v_covs_.push_back(Sigma_measure);
        Eigen::Matrix3d cov_reg = state_rot_ * Sigma_total * state_rot_.transpose();        // need extrinsics. But not implemented yet.
        v_covs_register_.push_back(cov_reg);
        v_cov_valid_.push_back(true);
    }
}

void PointCloudUn::calcMeasurementCov(PointType &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov){
    if (pb.y == 0) pb.y = 0.0001;
    float range = sqrt(pb.x * pb.x + pb.y * pb.y + pb.z * pb.z);
    float range_var = range_inc * range_inc;
    Eigen::Matrix2d direction_var;
    direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0, pow(sin(DEG2RAD(degree_inc)), 2);
    Eigen::Vector3d direction = Eigen::Vector3d(pb.x, pb.y, pb.z);
    direction.normalize();
    Eigen::Matrix3d direction_hat;
    direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
    Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
    base_vector1.normalize();
    Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
    base_vector2.normalize();
    Eigen::Matrix<double, 3, 2> N;
    N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
    Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
    cov = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
}



