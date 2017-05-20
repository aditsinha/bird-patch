
#include <fstream>
#include <cassert>
#include <vector>
#include <iostream>
#include <set>

#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImathBox.h>
#include <OpenEXR/ImfChannelList.h>

#include "io.h"

cv::Mat toBGR(cv::Mat usml) {
  // input color should be in UV, blue, red, green order
  int scale = 1 << 14;

  cv::Mat bgr(usml.size(), CV_8UC3);
  for (int i = 0; i < bgr.rows; i++) {
    for (int j = 0; j < bgr.cols; j++) {
      auto c = usml.at<cv::Vec4f>(i,j);

      bgr.at<cv::Vec3b>(i,j) = cv::Vec3b(c[1]*scale, c[2]*scale, c[3]*scale);
    }
  }
  return bgr;
}

cv::Mat toUSML(cv::Mat bgr) {
  // input color should be in UV, blue, red, green order
  int scale = 1 << 14;

  cv::Mat usml(bgr.size(), CV_32FC4);
  for (int i = 0; i < bgr.rows; i++) {
    for (int j = 0; j < bgr.cols; j++) {
      auto c = bgr.at<cv::Vec3b>(i,j);
      usml.at<cv::Vec4f>(i,j) = cv::Vec4f(0, (float)c[0]/scale, (float)c[1]/scale, (float)c[2]/scale);
    }
  }

  return usml;
}

void read_config_file(const std::string& config_filename, std::string* mesh_filename, std::string* texture_filename) {
  std::ifstream config(config_filename, std::ios_base::in);

  std::getline(config, *mesh_filename);
  std::getline(config, *texture_filename);
}

ObjData read_obj_file(const std::string& filename) {
  std::ifstream file(filename, std::ios_base::in);

  std::vector<GeoVert> geo;
  std::vector<TextCoord> text;

  ObjData data;

  int i = 0;

  while (file.good()) {
    i++;
    std::string line_type;
    file >> line_type;
    if (line_type[0] == '#') {
      // comment
      continue;
    } else if (line_type == "v") {
      // geometric vertex
      double x, y, z;
      file >> x >> y >> z;
      geo.emplace_back(x,y,z);
    } else if (line_type == "vt") {
      // texture coordinate
      double u, v;
      file >> u >> v;
      text.emplace_back(u,v);
    } else if (line_type == "f") {
      // face
      Face f;
      std::string line;
      std::getline(file, line);
      std::replace(line.begin(), line.end(), '/', ' ');
      std::istringstream is_line(line);
      is_line >> f.geoVert[0] >> f.textCoord[0] >> f.geoVert[1] >> f.textCoord[1] >> f.geoVert[2] >> f.textCoord[2];
      f.geoVert[0] -= 1;
      f.geoVert[1] -= 1;
      f.geoVert[2] -= 1;
      f.textCoord[0] -= 1;
      f.textCoord[1] -= 1;
      f.textCoord[2] -= 1;
      
      f.geoMat.row(0) = geo[f.geoVert[0]];
      f.geoMat.row(1) = geo[f.geoVert[1]];
      f.geoMat.row(2) = geo[f.geoVert[2]];
      
      data.faces.push_back(f);
    }
  }

  data.geo_verts = Eigen::MatrixX3d(geo.size(), 3);

  for (uint i = 0; i < geo.size(); i++) {
    data.geo_verts.block<1,3>(i, 0) = geo[i];
  }

  data.text_coords = Eigen::MatrixX2d(text.size(), 2);
  for (uint i = 0; i < text.size(); i++) {
    data.text_coords.block<1,2>(i, 0) = text[i];
  }

  return data;
}


cv::Mat read_texture(const std::string& filename) {
  Imf::InputFile file(filename.c_str());
  Imath::Box2i dw = file.header().dataWindow();

  int width = dw.max.x - dw.min.x + 1;
  int height = dw.max.y - dw.min.y + 1;
  std::cout << width << " " << height << std::endl;

  const Imf::ChannelList& channels = file.header().channels();
  std::set<std::string> channel_names;
  for (auto i = channels.begin(); i != channels.end(); ++i) {
    channel_names.insert(i.name());
  }

  // Read every channel in the image
  Imf::FrameBuffer frameBuffer;

  // read using the OpenEXR API into an OpenCV mat
  cv::Mat data(height, width, CV_32FC4);

  frameBuffer.insert("texture.lamp.usmlAvgBird.U",
  		     Imf::Slice(Imf::FLOAT,
  				(char*) data.ptr(),
				4*sizeof(float),
				4*sizeof(float) * width,
				1, 1, 0.0));
  frameBuffer.insert("texture.lamp.usmlAvgBird.S",
  		     Imf::Slice(Imf::FLOAT,
  				(char*) data.ptr() + sizeof(float),
				4*sizeof(float),
				4*sizeof(float) * width,
				1, 1, 0.0));
  frameBuffer.insert("texture.lamp.usmlAvgBird.M",
  		     Imf::Slice(Imf::FLOAT,
  				(char*) data.ptr() + 2*sizeof(float),
				4*sizeof(float),
				4*sizeof(float) * width,
				1, 1, 0.0));
  frameBuffer.insert("texture.lamp.usmlAvgBird.L",
  		     Imf::Slice(Imf::FLOAT,
  				(char*) data.ptr() + 3*sizeof(float),
				4*sizeof(float),
				4*sizeof(float) * width,
				1, 1, 0.0));

  file.setFrameBuffer(frameBuffer);
  file.readPixels(dw.min.y, dw.max.y);

  return data;
}

cv::Point2f translateToImg(float x, float y) {
  cv::Point2f pt((x+1)/2 * PROJECTION_PIXELS, (y+1)/2 * PROJECTION_PIXELS);
  assert(pt.x >= 0);
  assert(pt.y >= 0);
  return pt;
}

void draw_cluster(cv::Mat image, Eigen::MatrixX2d locations, cv::Scalar color) {
  for (int i = 0; i < locations.rows(); i++) {
    
    // cv::Point center((locations(i,0) + 1)/2 * PROJECTION_PIXELS,
    // 		     (locations(i,1) + 1)/2 * PROJECTION_PIXELS);

    cv::circle(image, translateToImg(locations(i,0), locations(i,1)), 2, color, -1);
  }
}

void draw_contour(cv::Mat image, std::vector<cv::Point2f> points, cv::Scalar color) {
  // we need to translate the points first
  std::vector<cv::Point> translated;

  for (const auto& pt : points) {
    translated.push_back(translateToImg(pt.x, pt.y));
  }

  std::vector<std::vector<cv::Point>> temp;
  temp.push_back(translated);

  cv::drawContours(image, temp, 0, color, 2);
}

void draw_contour2(cv::Mat image, std::vector<Eigen::Vector2d> points, cv::Scalar color) {
  // we need to translate the points first
  std::vector<cv::Point2f> cv_points;

  for (auto point : points) {
    cv_points.push_back({(float)point(0), (float)point(1)});
  }

  draw_contour(image, cv_points, color);
}

void colorize_clustering(cv::Mat image_out, cv::Mat image_in, std::set<int> clusters, std::set<int> borders) {
  cv::Mat in_hsv = cv::Mat::zeros(image_in.size(), CV_8UC3);

  std::map<int, cv::Vec3b> color_map;

  // Colors for the clusters
  int num_colors = clusters.size();
  int hue_change = 180 / num_colors;
  int i = 0;

  for (auto cluster : clusters) {
    std::cout << "Cluster: " << cluster << "\n";
    color_map[cluster] = cv::Vec3b(hue_change * i, 255, 255);
    i++;
  }

  // Colors for the borders
  for (auto border : borders) {
    color_map[border] = cv::Vec3b(0,128,128);
  }

  // colors for empty part
  color_map[0] = cv::Vec3b(0,0,0);

  for (int i = 0; i < image_in.cols; i++) {
    for (int j = 0; j < image_in.rows; j++) {
      int p = image_in.at<int>(cv::Point(i,j));
      in_hsv.at<cv::Vec3b>(cv::Point(i,j)) = color_map[p];
    }
  }

  cv::cvtColor(in_hsv, in_hsv, CV_HSV2BGR);
  cv::resize(in_hsv, image_out, image_out.size());
}
