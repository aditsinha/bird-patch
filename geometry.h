
#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

void center_mesh(ObjData* data);

std::vector<Face> get_front_faces(Eigen::MatrixX3d rotated_verts, const ObjData& mesh);

Eigen::MatrixX3d rotate_vertices(const ObjData& mesh, Eigen::Vector3d rotation);

Eigen::MatrixXd orthographic_sampling(const ObjData& mesh, const cv::Mat& texture, Eigen::MatrixX3d rotated_verts, std::vector<Face> front_faces);

cv::Mat orthographic_projection(const ObjData& mesh, const cv::Mat& texture, Eigen::MatrixX3d rotated_verts, std::vector<Face> front_faces);
