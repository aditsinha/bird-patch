
#include <fstream>
#include <cassert>
#include <vector>
#include <iostream>
#include <set>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

int main() {
  Eigen::Vector3d dir = Eigen::Vector3d::Ones();
  dir.normalize();
  
  double phi = acos(dir(2)), theta = atan2(dir(1), dir(0));
  std::cout << "Rotation" << std::endl;
  std::cout << phi << " " << theta << std::endl;

  // Eigen::Matrix3d rot =
  //   
  //   Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()).matrix();
  Eigen::Matrix3d rot =
    Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()).matrix() *
    Eigen::AngleAxisd(phi, Eigen::Vector3d::UnitY()).matrix();
  
  auto test_vec = Eigen::Vector3d::Ones().transpose();
  std::cout << test_vec * rot << std::endl;
}
