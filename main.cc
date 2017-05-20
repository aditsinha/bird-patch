
#include <fstream>
#include <cassert>
#include <vector>
#include <iostream>
#include <set>

#include <omp.h>

#include "common.h"
#include "io.h"
#include "geometry.h"
#include "clustering.h"

#define COLOR_LABEL_COUNT 4

int main(int argc, char** argv) {
  assert(argc == 2);

  std::string mesh_filename, texture_filename;
  read_config_file(argv[1], &mesh_filename, &texture_filename);
  auto texture = read_texture(texture_filename);
  // auto bgr_texture = cv::imread(texture_filename);
  // auto texture = toUSML(bgr_texture);

  auto mesh = read_obj_file(mesh_filename);
  center_mesh(&mesh);

  auto rotated_verts = rotate_vertices(mesh, -Eigen::Vector3d::UnitZ());
  auto front_faces = get_front_faces(rotated_verts, mesh);

  auto proj = orthographic_projection(mesh, texture, rotated_verts, front_faces);
  cv::imwrite("test.png", toBGR(proj));

  auto sampling = orthographic_sampling(mesh, texture, rotated_verts, front_faces);
  std::cout << "sampling size " << sampling.rows() << std::endl;
  auto labels = k_means_clustering(sampling, COLOR_LABEL_COUNT);
  auto sampling_divided = divide_by_clustering(sampling, labels);

  cv::Scalar colors[] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {255, 0, 255}, {0, 255, 255}, {255, 255, 255}};
    
  cv::Mat image = cv::Mat::zeros(PROJECTION_PIXELS, PROJECTION_PIXELS, CV_8UC3);

  std::vector<std::vector<Eigen::Vector2d>> color_contours;

  for (int i = 0; i < COLOR_LABEL_COUNT; i++) {
    auto sampling_with_density = add_density_data(sampling_divided[i].leftCols(2));
    sampling_with_density.block(0,2, sampling_with_density.rows(), 1) *= .2;

    auto density_labels = dbscan_clustering(sampling_with_density, .02, 20);

    auto density_divided = divide_by_clustering(sampling_divided[i], density_labels);

    for (const auto& d : density_divided) {
      auto hull = get_convex_hull(d);
      auto shape = get_alpha_shape_contours(d.leftCols(2), 10);
      if (shape.size() > 0) {
    	color_contours.push_back(shape[0]);
	draw_contour2(image, shape[0], colors[i % 7]);
      }
    }
  }

  // auto clusters = get_clusters(color_contours);

  // colorize_clustering(image, std::get<0>(clusters), std::get<1>(clusters), std::get<2>(clusters));

  cv::imwrite("out.png", image);
}
