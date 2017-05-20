
#pragma once

#include <eigen3/Eigen/Dense>

#define PROJECTION_PIXELS 2048

typedef Eigen::Vector3d GeoVert;
typedef Eigen::Vector2d TextCoord;

struct Face {
  int geoVert[3], textCoord[3];
  Eigen::Matrix3d geoMat;
};

struct ObjData {
  Eigen::MatrixX3d geo_verts;
  Eigen::MatrixX2d text_coords;
  std::vector<Face> faces;
};
