
#include "clustering.h"

#include <list>
#include <vector>
#include <functional>

#include <mlpack/methods/kmeans/kmeans.hpp>
#include <mlpack/methods/dbscan/dbscan.hpp>

#include <opencv2/opencv.hpp>

#define CLUSTERING_PIXELS 4096
#define CLUSTER_BORDER_THRESHOLD .1

Eigen::VectorXi k_means_clustering(Eigen::MatrixXd features, size_t num_clusters) {
  // we need to convert to Armadillo, because that is what the library
  // uses. the data must be column major
  arma::mat feature_mat(features.rows(), features.cols());
  for (int i = 0; i < features.rows(); i++) {
    for (int j = 0; j < features.cols(); j++) {
      feature_mat(i,j) = features(i,j);
    }
  }

  arma::inplace_trans(feature_mat);
  // arma::mat means;
  // bool status = arma::kmeans(means, feature_mat, num_clusters, arma::random_subset, 10, true);

  arma::Row<size_t> assignments;
  mlpack::kmeans::KMeans<> k;
  k.Cluster(feature_mat, num_clusters, assignments);

  Eigen::VectorXi result(assignments.n_elem);
  for (int i = 0; i < assignments.n_elem; i++) {
    result(i) = assignments(i);
  }

  return result;
}

Eigen::VectorXi dbscan_clustering(Eigen::MatrixXd features, double epsilon, int min_points) {
  arma::mat feature_mat(features.rows(), features.cols());
  for (int i = 0; i < features.rows(); i++) {
    for (int j = 0; j < features.cols(); j++) {
      feature_mat(i,j) = features(i,j);
    }
  }

  arma::inplace_trans(feature_mat);

  arma::Row<size_t> assignments;
  mlpack::dbscan::DBSCAN<> dbscan(epsilon, min_points);
  dbscan.Cluster(feature_mat, assignments);

  Eigen::VectorXi result(assignments.n_elem);
  for (int i = 0; i < assignments.n_elem; i++) {
    result(i) = assignments(i);
  }

  return result;
}

std::vector<Eigen::MatrixXd> divide_by_clustering(Eigen::MatrixXd data, Eigen::VectorXi labels) {
  int num_clusters = labels.maxCoeff() + 1;

  std::vector<Eigen::MatrixXd> clusters;
  clusters.resize(num_clusters);

  for (int i = 0; i < num_clusters; i++) {
    // create a matrix for the cluster
    int cluster_size = (labels.array() == i).count();
    clusters[i] = Eigen::MatrixXd::Zero(cluster_size, data.cols());

    // copy rows into the cluster matrix
    int r = 0;
    for (int j = 0; j < data.rows(); j++) {
      if (labels(j) == i) {
	clusters[i].row(r) = data.row(j);
	r++;
      }
    }
  }

  return clusters;
}

cv::Point2f translateToClusterImg(float x, float y) {
  cv::Point2f pt((x+1)/2 * CLUSTERING_PIXELS, (y+1)/2 * CLUSTERING_PIXELS);
  assert(pt.x >= 0);
  assert(pt.y >= 0);
  return pt;
}

Eigen::MatrixX3d add_density_data(Eigen::MatrixX2d cluster) {
  // first draw the outline onto a blank image
  Eigen::MatrixX3d result(cluster.rows(), 3);
  
  cv::Mat canvas = cv::Mat::zeros(CLUSTERING_PIXELS, CLUSTERING_PIXELS, CV_8UC1);

  // draw each point
  for (int i = 0; i < cluster.rows(); i++) {
    cv::circle(canvas, translateToClusterImg(cluster(i,0), cluster(i,1)), 1, cv::Scalar(128), -1);
  }

  // now go through each point and get density from a small neighborhood around each point
  for (int i = 0; i < cluster.rows(); i++) {
    int sum = 0;
    auto base = translateToClusterImg(cluster(i,0), cluster(i,1));
    for (int j = -7; j <= 7; j++) {
      for (int k = -7; k <= 7; k++) {
	auto loc = base;
	loc.x += j; loc.y += k;
	if (loc.x < 0 || loc.x > CLUSTERING_PIXELS || loc.y < 0 || loc.y > CLUSTERING_PIXELS) {
	  continue;
	}
	
	sum += canvas.at<uchar>(loc);
      }
    }

    result(i, 0) = cluster(i, 0);
    result(i, 1) = cluster(i, 1);
    result(i, 2) = ((double)sum) / 128 / 49;
  }

  return result;
}

std::vector<cv::Point2f> get_convex_hull(Eigen::MatrixXd cluster) {
  std::vector<cv::Point2f> in_vec, out_vec;

  for (int i = 0; i < cluster.rows(); i++) {
    in_vec.emplace_back(cluster(i,0), cluster(i,1));
  }

  cv::convexHull(in_vec, out_vec);
  return out_vec;
}


bool has_empty_disk(Eigen::Vector2d p1, Eigen::Vector2d p2, std::vector<Eigen::Vector2d> candidates, double radius) {
  auto p_vec = p1 - p2;
  double p_dist = p_vec.norm();
  
  auto midpoint = (p1+p2)/2;
  
  Eigen::Vector2d normal;
  normal(0) = -p_vec(1); normal(1) = p_vec(0);
  normal.normalize();

  double offset = sqrt(pow(radius, 2) - pow(p_dist/2, 2));

  auto disk1_c = midpoint + normal * offset;
  auto disk2_c = midpoint - normal * offset;

  bool disk1_empty = true, disk2_empty = true;

  for (auto candidate : candidates) {
    if ((candidate - disk1_c).squaredNorm() <= radius*radius) {
      disk1_empty = false;
    }

    if ((candidate - disk2_c).squaredNorm() <= radius*radius) {
      disk2_empty = false;
    }
  }

  return disk1_empty || disk2_empty;
}

double circumcircle_radius(Eigen::Vector2d p1, Eigen::Vector2d p2, Eigen::Vector2d p3) {
  // first calculate the length of each side.
  double a = (p1 - p2).norm(), b = (p2 - p3).norm(), c = (p3 - p1).norm();

  // the semiperimeter
  double s = (a + b + c) / 2;

  // area by Heron's formula
  double A = sqrt(s * (s-a) * (s-b) * (s-c));

  // radius is (abc / 4A)

  return (a * b * c) / (4 * A);
}

std::vector<Eigen::Vector2d> indices_to_points(Eigen::MatrixX2d points, std::vector<int> indices) {
  std::vector<Eigen::Vector2d> vec;

  for (int i : indices) {
    vec.push_back(points.row(i));
  }

  return vec;
}

std::vector<std::vector<Eigen::Vector2d>> get_alpha_shape_contours(Eigen::MatrixX2d cluster, double alpha) {

  // by Wikipedia convention, setting alpha = 0 results in a convex hull.
  double inv_alpha = 1 / alpha;
  
  // first compute the Delaunay triangulation
  cv::Subdiv2D subdiv(cv::Rect(0, 0, CLUSTERING_PIXELS, CLUSTERING_PIXELS));

  std::map<std::pair<float, float>, int> index_map;

  for (int i = 0; i < cluster.rows(); i++) {
    auto trans_point = translateToClusterImg(cluster(i,0), cluster(i,1));
    subdiv.insert(trans_point);
    index_map[{trans_point.x, trans_point.y}] = i;
  }

  std::vector<cv::Vec4f> edge_list;
  subdiv.getEdgeList(edge_list);

  int num_edges = edge_list.size();

  // calculate adjacency sets for each vertex
  std::vector<std::set<int>> adjacency_sets;

  std::list<std::pair<int,int>> alpha_shape_edges;
  
  for (int i = 0; i < cluster.rows(); i++) {
    adjacency_sets.push_back(std::set<int>());
  }

  for (cv::Vec4f e : edge_list) {
    if (index_map.count({e[0], e[1]}) == 0 || index_map.count({e[2], e[3]}) == 0) {
      // border edge added by openCV
      continue;
    }

    int idx0 = index_map[{e[0], e[1]}];
    int idx1 = index_map[{e[2], e[3]}];

    adjacency_sets[idx0].insert(idx1);
    adjacency_sets[idx1].insert(idx0);
  }

  // now iterate through all edges in the triangulation and see which
  // edges should be in the alpha-shape
  for (int i = 0; i < cluster.rows(); i++) {
    const auto& adj_set = adjacency_sets[i];
    for (auto j : adj_set) {
      if (j < i) {
	// don't double process nodes
	continue;
      }
      
      // ensure that these points are close enough.
      if ((cluster.row(i) - cluster.row(j)).norm() >= 2*inv_alpha) {
	// too far away to be considered.
	continue;
      }

      std::vector<int> intersection;
      
      // figure out the third vertex in the triangle(s) containing i and
      // j.
      std::set_intersection(adj_set.begin(), adj_set.end(), adjacency_sets[j].begin(), adjacency_sets[j].end(),
			    std::back_inserter(intersection));

      // in order to be in a triangle, there must have been at least one other node

      if (intersection.size() == 0) {
	continue;
      }
      
      // assert(0 < intersection.size());// && intersection.size() < 3);

      std::vector<Eigen::Vector2d> close_points;
      for (auto k : intersection) {
	close_points.push_back(cluster.row(k));
      }

      if (has_empty_disk(cluster.row(i), cluster.row(j), close_points, inv_alpha)) {
	alpha_shape_edges.emplace_front(i, j);
      }
    }
  }

  // convert the alpha shape edges into contours
  std::vector<std::vector<int>> contours;

  while (!alpha_shape_edges.empty()) {
    std::vector<int> contour;
    // start the contour with the head of the list
    auto first_edge = alpha_shape_edges.front();
    int first = std::get<0>(first_edge), last = std::get<1>(first_edge);
    
    contour.push_back(first);
    contour.push_back(last);

    alpha_shape_edges.pop_front();

    bool found_contour = false;
    while (!found_contour) {
      // repeatedly iterate over list to find contour
      int did_make_change = false;
      auto it = alpha_shape_edges.begin();
      while (it != alpha_shape_edges.end()) {
	auto edge = *it;
	int edge_a = std::get<0>(edge), edge_b = std::get<1>(edge);
	if (edge_b == last) {
	  // swap a and b so that a becomes last and we only need to
	  // write the following logic once
	  int temp = edge_a;
	  edge_a = edge_b;
	  edge_b = temp;
	}
      
	if (edge_a == last) {
	  // we are going to use this edge
	  it = alpha_shape_edges.erase(it);
	  if (edge_b == first) {
	    // close the contour
	    found_contour = true;
	    did_make_change = true;
	    break;
	  } else {
	    // otherwise continue it
	    contour.push_back(edge_b);
	    last = edge_b;
	    did_make_change = true;
	  }
	} else {
	  ++it;
	}
      }

      if (!did_make_change) {
	// unable to find this contour. assume it is invalid
	break;
      }
    }

    if (found_contour) {
      // we have a full contour
      contours.push_back(contour);
    }
  }

  std::vector<std::vector<Eigen::Vector2d>> result;
  std::transform(contours.begin(), contours.end(), std::back_inserter(result),
		 std::bind(indices_to_points, cluster, std::placeholders::_1));

  return result;
}


std::tuple<cv::Mat, std::set<int>, std::set<int>>
  get_clusters(std::vector<std::vector<Eigen::Vector2d>> color_contours) {
  
  // handle more contours later
  assert(color_contours.size() < 32);

  // draw each of the clusters onto its own canvas
  std::vector<cv::Mat> contour_canvases;
  std::vector<int> contour_area;
  for (const auto& cc : color_contours) {
    // translate all of the contours to opencv points
    std::vector<cv::Point> translated;
    for (const auto& p : cc) {
      translated.push_back(translateToClusterImg(p(0), p(1)));
    }
    std::vector<std::vector<cv::Point>> temp;
    temp.push_back(translated);
    
    cv::Mat canvas = cv::Mat::zeros(CLUSTERING_PIXELS, CLUSTERING_PIXELS, CV_32SC1);

    cv::drawContours(canvas, temp, 0, {1}, -1);
    int area = cv::countNonZero(canvas);
    std::cout << "Contour: " << area << "\n";
    contour_area.push_back(area);
    contour_canvases.push_back(canvas);
  }

  cv::Mat combined = cv::Mat::zeros(CLUSTERING_PIXELS, CLUSTERING_PIXELS, CV_32SC1);
  // treat this as a 31 bit unsigned integer, where we shift and add
  // each time
  for (int i = contour_canvases.size() - 1; i >= 0; i--) {
    combined = combined*2;
    combined += contour_canvases[i];
  }

  // count all of the different clusters that show up along with the areas
  std::set<int> possible_clusters;

  for (int i = 0; i < CLUSTERING_PIXELS; i++) {
    for (int j = 0; j < CLUSTERING_PIXELS; j++) {
      int p = combined.at<int>(i,j);
      if (p != 0) {
	possible_clusters.insert(p);
      }
    }
  }

  std::set<int> clusters, borders;

  // for each possible_cluster, determine the area
  for (int possible_cluster : possible_clusters) {
    int area = cv::countNonZero(combined == possible_cluster);
    // figure out whether this is a full cluster or a border region
    bool is_border = true;

    std::cout << "Patch: " << area << "\n";
    for (int i = 0; i < 31; i++) {
      if (possible_cluster & (1 << i)) {
	// this is one of the contours in the cluster
	if (contour_area[i] * CLUSTER_BORDER_THRESHOLD < area) {
	  // the area of this cluster is big enough that it's not considered a border of this cluster
	  is_border = false;
	  break;
	}
      }
    }

    if (is_border) {
      borders.insert(possible_cluster);
    } else {
      clusters.insert(possible_cluster);
    }
  }

  return std::make_tuple(combined, clusters, borders);
}
