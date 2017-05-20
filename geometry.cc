
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "common.h"
#include "geometry.h"

#define FACE_SAMPLING 5

void center_mesh(ObjData* data) {
  // center the geometric vertices around the origin and scale them so
  // that they fall into the ball of radius 1, without distorting the image.
  auto min = data->geo_verts.colwise().minCoeff();
  auto max = data->geo_verts.colwise().maxCoeff();

  auto avg = (min + max) / 2;

  data->geo_verts.rowwise() -= avg;

  auto max_norm = data->geo_verts.rowwise().norm().maxCoeff();
  double scale = 1 / max_norm;

  data->geo_verts *= scale;

  std::cout << "Min: " << data->geo_verts.colwise().minCoeff() << std::endl;
  std::cout << "Max: " << data->geo_verts.colwise().maxCoeff() << std::endl;
  std::cout << "Scale: " << scale << std::endl;
}

Eigen::MatrixX3d rotate_vertices(const ObjData& mesh, Eigen::Vector3d rotation) {
  Eigen::Vector3d dir = rotation.normalized();
  
  // the following rotation matricies create a rotation to the camera
  // direction.  They are derived from converting the camera direction
  // to polar coordinates, then treating the camera's position as a
  // rotation around the Y-axis then around the Z-axis.

  double phi = acos(dir(2)), theta = atan2(dir(1), dir(0));
  std::cout << "Rotation" << std::endl;
  std::cout << phi << " " << theta << std::endl;
  
  Eigen::Matrix3d rot =
    Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()).matrix() *
    Eigen::AngleAxisd(phi, Eigen::Vector3d::UnitY()).matrix();

  auto rotated_verts = mesh.geo_verts * rot;
  return rotated_verts;
}

std::vector<Face> get_front_faces(Eigen::MatrixX3d rotated_verts, const ObjData& mesh) {
  std::vector<Face> sampled_faces;
  double sample_threshold = RAND_MAX * 1./FACE_SAMPLING;
  for (int i = 0; i < mesh.faces.size(); i++) {
    if (rand() < sample_threshold) {
      sampled_faces.push_back(mesh.faces[i]);
    }
  }

  std::vector<Face> front_faces;
  std::vector<std::vector<Face>> ff_buffers;

#pragma omp parallel
  {
    auto nthreads = omp_get_num_threads();
    auto id = omp_get_thread_num();
    auto num_faces = sampled_faces.size();

#pragma omp single
    {
      ff_buffers.resize(nthreads);
    }
#pragma omp for
    for (int i = 0; i < num_faces; i++) {
      auto v0 = Eigen::Vector3d(rotated_verts.row(sampled_faces[i].geoVert[0]));
      auto v1 = Eigen::Vector3d(rotated_verts.row(sampled_faces[i].geoVert[1]));
      auto v2 = Eigen::Vector3d(rotated_verts.row(sampled_faces[i].geoVert[2]));

      auto normal = (v1 - v0).cross(v2 - v0);

      // the viewpoint for this face should be above it in the z direction
      if (normal(2) > 0) {
	ff_buffers[id].push_back(sampled_faces[i]);
      }
    }

    std::cout << "Done finding front faces" << std::endl;

#pragma omp single
    {
      for (auto& buffer : ff_buffers) {
	std::move(buffer.begin(), buffer.end(), std::back_inserter(front_faces));
      }
    }
  }

  return front_faces;
}

Eigen::MatrixXd orthographic_sampling(const ObjData& mesh, const cv::Mat& texture, Eigen::MatrixX3d rotated_verts, std::vector<Face> front_faces) {

  Eigen::MatrixXd feature_matrix(front_faces.size(), 6);

  int count = 0;
  int i = 0;
  for (const auto& face : front_faces) {

    if (count % 1000 == 0) {
      std::cout << "Face: " << count << std::endl;
    }

    std::vector<cv::Point2f> src_tri, dst_tri;
    Eigen::VectorXd features = Eigen::VectorXd::Zero(6);

    for (int i = 0; i < 3; i++) {
      auto texture_coord = Eigen::Vector2d(mesh.text_coords.row(face.textCoord[i]));
      // we need to flip the Y-coordinate because the origin is in the
      // bottom left in UV while it is in the top left for OpenCV.
      src_tri.emplace_back(texture.cols * texture_coord(0), texture.rows * (1 - texture_coord(1)));

      auto geo_coord = rotated_verts.row(face.geoVert[i]);
      // figure out the location on the image that this triangle should be drawn
      dst_tri.emplace_back((geo_coord[0] + 1)*PROJECTION_PIXELS/2, (geo_coord[1] + 1)*PROJECTION_PIXELS/2);

      features(0) += geo_coord[0];
      features(1) += geo_coord[1];
    }

    // features now contains the centroid of the mesh triangle
    features /= 3;

    auto src_rect = cv::boundingRect(src_tri), dst_rect = cv::boundingRect(dst_tri);
    cv::Mat src_cropped, dst_cropped = cv::Mat::zeros(dst_rect.size(), texture.type());
    texture(src_rect).copyTo(src_cropped);

    std::vector<cv::Point2f> src_tri_crop, dst_tri_crop;
    std::vector<cv::Point> dst_tri_crop_int;
    double side_len[3];
    
    for (int i = 0; i < 3; i++) {
      src_tri_crop.emplace_back(src_tri[i].x - src_rect.x, src_tri[i].y - src_rect.y);
      dst_tri_crop.emplace_back(dst_tri[i].x - dst_rect.x, dst_tri[i].y - dst_rect.y);
      dst_tri_crop_int.emplace_back(dst_tri[i].x - dst_rect.x, dst_tri[i].y - dst_rect.y);

      side_len[i] = cv::norm(dst_tri[i] - dst_tri[(i+1)%3]);
    }

    double semi_perim = (side_len[0] + side_len[1] + side_len[2])/2;
    // use Heron's formula to get area
    double dst_area = sqrt(semi_perim *
			   (semi_perim - side_len[0]) *
			   (semi_perim - side_len[1]) *
			   (semi_perim - side_len[2]));

    // do the affine transformation
    cv::Mat transform = cv::getAffineTransform(src_tri_crop, dst_tri_crop);
    cv::warpAffine(src_cropped, dst_cropped, transform, dst_cropped.size());
    
    // mask everything not in the desired triangle
    cv::Mat mask = cv::Mat::zeros(dst_cropped.size(), dst_cropped.type());
    cv::fillConvexPoly(mask, dst_tri_crop_int, cv::Scalar(1, 1, 1, 1));

    cv::multiply(dst_cropped, mask, dst_cropped);

    if (dst_area > 1) {
      // figure out the average of each channel over the triangle
      auto average_color = (dst_area > 0) ? cv::sum(dst_cropped) / dst_area : 0;
      average_color *= 1 << 12;
      features(2) = average_color[0] * 1;
      features(3) = average_color[1] * 1;
      features(4) = average_color[2] * 1;
      features(5) = average_color[3] * 1;
      // std::cout << average_color << std::endl;
      feature_matrix.row(i) = features;
      i++;
    }

    count++;
  }

  return feature_matrix.block(0, 0, i, 6);
}


cv::Mat orthographic_projection(const ObjData& mesh, const cv::Mat& texture, Eigen::MatrixX3d rotated_verts, std::vector<Face> front_faces) {
  cv::Mat image = cv::Mat::zeros(PROJECTION_PIXELS, PROJECTION_PIXELS, CV_32FC4);

  std::cout << "Front faces: " << front_faces.size() << std::endl;

  int count = 0;

  for (const auto& face : front_faces) {
    count++;
    if (count % 1000 == 0) {
      std::cout << "Face: " << count << std::endl;
    }
    
    std::vector<cv::Point2f> src_tri, dst_tri;

    for (int i = 0; i < 3; i++) {
      auto texture_coord = Eigen::Vector2d(mesh.text_coords.row(face.textCoord[i]));
      // we need to flip the Y-coordinate because the origin is in the
      // bottom left in UV while it is in the top left for OpenCV.
      src_tri.emplace_back(texture.cols * texture_coord(0), texture.rows * (1 - texture_coord(1)));

      auto geo_coord = rotated_verts.row(face.geoVert[i]);
      // figure out the location on the image that this triangle should be drawn
      dst_tri.emplace_back((geo_coord[0] + 1)*PROJECTION_PIXELS/2, (geo_coord[1] + 1)*PROJECTION_PIXELS/2);
    }

    auto src_rect = cv::boundingRect(src_tri), dst_rect = cv::boundingRect(dst_tri);
    cv::Mat src_cropped, dst_cropped = cv::Mat::zeros(dst_rect.size(), texture.type());

    texture(src_rect).copyTo(src_cropped);

    std::vector<cv::Point2f> src_tri_crop, dst_tri_crop;
    std::vector<cv::Point> dst_tri_crop_int;
    for (int i = 0; i < 3; i++) {
      src_tri_crop.emplace_back(src_tri[i].x - src_rect.x, src_tri[i].y - src_rect.y);
      dst_tri_crop.emplace_back(dst_tri[i].x - dst_rect.x, dst_tri[i].y - dst_rect.y);
      dst_tri_crop_int.emplace_back(dst_tri[i].x - dst_rect.x, dst_tri[i].y - dst_rect.y);
    }

    // do the affine transformation
    cv::Mat transform = cv::getAffineTransform(src_tri_crop, dst_tri_crop);
    cv::warpAffine(src_cropped, dst_cropped, transform, dst_cropped.size());
    
    // mask everything not in the desired triangle
    cv::Mat mask = cv::Mat::zeros(dst_cropped.size(), dst_cropped.type());
    cv::fillConvexPoly(mask, dst_tri_crop_int, cv::Scalar(1, 1, 1, 1));

    cv::multiply(dst_cropped, mask, dst_cropped);
    
    // copy the triangle to the entire image uncomment the following
    // line if you want to clear the existing triangle before applying
    // a new one.
    cv::multiply(image(dst_rect), cv::Scalar(1.0, 1.0, 1.0, 1.0) - mask, image(dst_rect));
    image(dst_rect) = image(dst_rect) + dst_cropped;
  }

  return image;
}
