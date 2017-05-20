
#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>
#include <opencv2/opencv.hpp>

Eigen::VectorXi k_means_clustering(Eigen::MatrixXd features, size_t num_clusters);
Eigen::VectorXi dbscan_clustering(Eigen::MatrixXd features, double epsilon, int min_points);
std::vector<Eigen::MatrixXd> divide_by_clustering(Eigen::MatrixXd data, Eigen::VectorXi labels);
std::vector<cv::Point2f> get_convex_hull(Eigen::MatrixXd cluster);
std::vector<std::vector<Eigen::Vector2d>> get_alpha_shape_contours(Eigen::MatrixX2d cluster, double alpha);
std::tuple<cv::Mat, std::set<int>, std::set<int>>
  get_clusters(std::vector<std::vector<Eigen::Vector2d>> color_contours);
Eigen::MatrixX3d add_density_data(Eigen::MatrixX2d cluster);
