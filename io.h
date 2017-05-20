
#pragma once

#include <opencv2/opencv.hpp>

#include "common.h"

cv::Mat toBGR(cv::Mat usml);
cv::Mat toUSML(cv::Mat bgr);

void read_config_file(const std::string& config_filename, std::string* mesh_filename, std::string* texture_filename);

ObjData read_obj_file(const std::string& filename);
cv::Mat read_texture(const std::string& filename);
void draw_cluster(cv::Mat image, Eigen::MatrixX2d locations, cv::Scalar color);
void draw_contour(cv::Mat image, std::vector<cv::Point2f> points, cv::Scalar color);
void draw_contour2(cv::Mat image, std::vector<Eigen::Vector2d>, cv::Scalar color);
void colorize_clustering(cv::Mat image_out, cv::Mat image_in, std::set<int> clusters, std::set<int> borders);
