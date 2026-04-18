#ifndef VOXEL_MAP_UTIL_HPP
#define VOXEL_MAP_UTIL_HPP
#include "common_lib.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <string>

#define HASH_P 116101
#define MAX_N 10000000000


class p2v{
public:
  p2v(void) {}
  p2v(V3D query_world) : query_world(query_world), is_valid(false) {}
  Eigen::Vector3d query_world;
  Eigen::Vector3d body;   // needed for residual calculation.
  Eigen::Vector3d vec;
  double sigma_d;          
  bool is_valid = false;
  void print(void) { 
    cout << "Print p2v class: query_world: [" << query_world.transpose() << " ], is_valid: " << is_valid 
    << ", vec: " << vec.transpose() << ", sigma d: " << sigma_d << endl; 
  }
};

struct UnifiedMeasurement{   
  V3D point;    // point not in "body" frame. Need extrinsics
  V3D world;    // world: point->IMU(body)->world
  V3D norm;     // norm direction: from world-point to voxel.
  double dist;  // world->normal distance (should always be positive).
  double R_inv; // 1/cov, R.
};

class VOXEL_LOC {
public:
  int64_t x, y, z;

  VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}
  
  bool operator==(const VOXEL_LOC &other) const {
    return (x == other.x && y == other.y && z == other.z);
  }
};

// Hash value
namespace std {
template <> struct hash<VOXEL_LOC> {
  int64_t operator()(const VOXEL_LOC &s) const {
    using std::hash;
    using std::size_t;
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};
} // namespace std


M3D calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc);


#endif
