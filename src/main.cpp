
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

// Boost Library
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

// OpenCV Library
#include <opencv2/opencv.hpp>

//
#include "stereo_colorize.h"
#include "stixel_generator.h"
#include "ground_estimator.h"
#include "hough_trans.h"

namespace fs = boost::filesystem;
using namespace koichi_robotics_lib;

bool dir_exist(std::string dir_path) {

  boost::system::error_code error;
  const bool result = fs::exists(dir_path, error);
  return result && !error;
}

void extract_img_pairs(std::string dir_path, std::vector<std::pair<std::string, std::string> > &img_pair) {

  const fs::path path(dir_path);
  std::vector<std::string> left_img, right_img;

  BOOST_FOREACH(const fs::path& p, std::make_pair(fs::directory_iterator(path), fs::directory_iterator())) {
    if (!fs::is_directory(p)) { 
      std::string imgpath = p.parent_path().generic_string() + "/"+ p.filename().generic_string();
      std::cout << imgpath << std::endl;      
      if (p.filename().generic_string().find("img_c1_") != std::string::npos) {
        left_img.push_back(imgpath);
      } else if (p.filename().generic_string().find("img_c0_") != std::string::npos) {
        right_img.push_back(imgpath);
      }
    }
  }

  std::sort(left_img.begin(), left_img.end());
  std::sort(right_img.begin(), right_img.end());

  for (int i = 0; i < left_img.size(); i++) {
    std::pair<std::string, std::string> pair(left_img[i], right_img[i]);
    img_pair.push_back(pair);    
  }

}

void convert_to_8U(std::string &left_path, std::string &right_path, cv::Mat &left_8, cv::Mat &right_8) {

  cv::Mat left_16 = cv::imread(left_path, cv::IMREAD_UNCHANGED);
  cv::Mat right_16 = cv::imread(right_path, cv::IMREAD_UNCHANGED);

  cv::normalize(left_16, left_16, 0, 255, cv::NORM_MINMAX);
  cv::normalize(right_16, right_16, 0, 255, cv::NORM_MINMAX);

  left_16.convertTo(left_8, CV_8U);
  right_16.convertTo(right_8, CV_8U);

}

void convert32F_to_8U(cv::Mat &org, cv::Mat &dst) {

  double min, max;
  cv::minMaxLoc(org, &min, &max);
  cv::convertScaleAbs(org, dst, 255 / (max - min), 255 / min);   

}

void generate_sgbm_obj(cv::Ptr<cv::StereoSGBM> &sgbm_ptr, int disp) {

  int min_disp = 0;
  int max_disp = disp;
  int block_size = 5;
  int P1 = 1000;
  int P2 = 3000;
  int disp12_max_diff = 0;
  int pre_filt_cap = 0;
  int uniquness_ratio = 0;
  int speckle_wnd_size = 0;
  int speckle_range = 0;
  int mode = cv::StereoSGBM::MODE_SGBM_3WAY;

  sgbm_ptr = cv::StereoSGBM::create(min_disp, max_disp, block_size, P1, P2, 
                           disp12_max_diff, pre_filt_cap, uniquness_ratio, speckle_wnd_size,
                           speckle_range, mode);
}

void prepare_stereo_colorize_object(StereoColorize &stereoColorize, int width, int height, int disp) {

  StereoColorizeParams params;
  params.grayScale = false;
  params.red2blue = true;
  params.width = width;
  params.height = height;
  params.min = 0;
  params.max = disp;

  stereoColorize.initialize(params);
}

void prepare_gnd_est_object(GroundEstimator &gndEst, int &width, int &height, int &disp) {

  GroundEstimatorParams params;
  params.v_disp_height = height;
  params.v_disp_width = disp;
  params.min_disp = 0;
  params.max_disp = disp;
  params.show_v_disp = true;

  params.hough_param.width = width;
  params.hough_param.height = height;
  params.hough_param.rho_max = static_cast<int>(std::sqrt(height*height + disp*disp));
  params.hough_param.theta_max = 180;
  params.hough_param.score_thresh = 10;
  params.hough_param.rho_res = 0.5;
  params.hough_param.theta_res = 0.25;
  params.hough_param.vote_thr_ratio_wrt_max = 0.3;
  params.hough_param.min_vote_per_pix = 1;
  params.hough_param.max_vote_per_pix = 255;
  params.hough_param.line_rho_min = 0.0 * params.hough_param.rho_max / params.hough_param.rho_res;
  params.hough_param.line_rho_max = 0.1 * params.hough_param.rho_max / params.hough_param.rho_res;
  params.hough_param.line_theta_min = 0.90 * params.hough_param.theta_max / params.hough_param.theta_res;
  params.hough_param.line_theta_max = 0.98 * params.hough_param.theta_max / params.hough_param.theta_res;

  gndEst.initialize(params);
}

void prepare_stixel_gen_obj(StixelGenerator &stixelGen, int &width, int &height, int &disp) {

  StixelGeneratorParams params;
  // Original Image
  params.img_width = width;
  params.img_height = height;
  // Disparity Image
  params.min_disp = 0;
  params.max_disp = disp;
  // Stixel
  params.stixelWidth = 5;
  params.max_depth = 100;
  params.depth_res = 0.25;
  // Cost Image
  params.obj_height = 0.20;
  params.below_under_gnd = 0.50;
  params.road_dev_cost = 1;
  params.obst_dev_cost = 10;
  //params.ignr_drow_low = 40;
  params.ignr_drow_low = 0;
  //params.ignr_drow_up = 80;
  params.ignr_drow_up = 0;

  // Dynamic Programming
  params.dv_horizon = -20;
  params.space_smooth_fac = 10.0;
  params.upper_spatial_dist = 100;

  // Camera
  params.camParam.f = 1245.0;
  params.camParam.u0 = 472.735;
  params.camParam.v0 = 175.787;
  params.camParam.baseline = 0.214382;
  params.camParam.height = 1.17;
  params.camParam.tilt_rad = 0.081276;

  stixelGen.initialize(params);

}

void get_img_size(std::string &path, int &width, int &height) {

  cv::Mat mat = imread(path, cv::IMREAD_UNCHANGED);
  width = mat.cols;
  height = mat.rows;
}

void remove_neg_value(cv::Mat &src_f32) {

  float * const pnt = reinterpret_cast<float *>(src_f32.data);
  for (int v = 0; v < src_f32.rows; v++) {
    for (int u = 0; u < src_f32.cols; u++) {
      float *tgt = pnt + src_f32.cols * v + u;
      if (*tgt == -1.0) {
        *tgt = 255.0f;
      }
    }
  }

}

void draw_fs_boundary(const std::vector<int> fs_boundary, cv::Mat &img) {

  img = 0;

  for (int u = 0; u < img.cols; u++) {
    cv::line(img, cv::Point(u, fs_boundary[u]), cv::Point(u, img.rows-1), cv::Scalar(0, 0, 200), 1,4);
  }

}

int main(int argc, char **argv) {
  
  std::cout << "Stixel Test" << std::endl;

  if (argc < 2 || !dir_exist(argv[1])) {
    std::cout << "Please input valid image directory." << std::endl;
    return 0;
  }

  std::vector<std::pair<std::string, std::string>> img_path;
  extract_img_pairs(argv[1], img_path);

  int width, height;
  int max_disp = 64;
  // Get Image information.
  std::cout << img_path.size() << std::endl;
  get_img_size(img_path[0].first, width, height);

  // Prepare SGBM Object
  cv::Ptr<cv::StereoSGBM> sgbm_ptr;
  generate_sgbm_obj(sgbm_ptr, max_disp);

  // Prepare Ground Estimator Object;
  GroundEstimator gndEst;
  prepare_gnd_est_object(gndEst, width, height, max_disp);

  // Prepare Stixel Generator Object.
  StixelGenerator stixelGen;
  prepare_stixel_gen_obj(stixelGen, width, height, max_disp);

  // Prepare Stereo Colorize Object
  StereoColorize stereoColorize;
  prepare_stereo_colorize_object(stereoColorize, width, height, max_disp);

  cv::Mat left_8, right_8, disp, disp_u8, v_disp, hough_vote_u8, u_disp, score_u8, u_disp_resized, u_disp_foregnd, u_disp_foregnd_resized;
  cv::Mat disp_rgb(height, width, CV_8UC3);
  cv::Mat free_spc(height, width, CV_8UC3);
  cv::Mat left_rgb(height, width, CV_8UC3);
  cv::Mat left_fused(height, width, CV_8UC4);

  for (auto itr = img_path.begin(); itr != img_path.end(); itr++) {
    convert_to_8U(itr->first, itr->second, left_8, right_8);
    cv::cvtColor(right_8, left_rgb, CV_GRAY2RGB);   

    sgbm_ptr->compute(right_8, left_8, disp);
    disp.convertTo(disp, CV_32F, 1.0 / cv::StereoSGBM::DISP_SCALE);
    remove_neg_value(disp);
    stereoColorize.colorizeFloat((float *)disp.data, disp_rgb.data);    

    // Road Estimation By Hough Transform on Disparity Image.
    std::vector<GroundModel> gnd_models;    
    disp.convertTo(disp_u8, CV_8U, 1, 0);
    gndEst.estimate_ground(disp_u8, gnd_models);
    gndEst.get_line_overlaid_img(v_disp);
    gndEst.get_hough_vote_img(hough_vote_u8);

    // Stixel Generation
    std::vector<Stixel> stixels;
    std::vector<int> fs_boundary;
    stixelGen.generate_stixel(disp_u8, gnd_models, stixels);
    stixelGen.get_u_disp_img(u_disp);
    stixelGen.get_u_disp_foregnd(u_disp_foregnd);
    cv::resize(u_disp, u_disp_resized, cv::Size(), 1.0, 5.5);
    cv::resize(u_disp_foregnd, u_disp_foregnd_resized, cv::Size(), 1.0, 5.5);
    stixelGen.get_score_img(score_u8);
    stixelGen.get_fs_boundary(fs_boundary);
    draw_fs_boundary(fs_boundary, free_spc);
    cv::addWeighted(left_rgb, 0.9, free_spc, 0.3, 0.0, left_fused);

    // Free Space Calculation.
    cv::imshow("Left", left_8);
    cv::imshow("Left with FS", left_fused);
    cv::imshow("Right", right_8);
    cv::imshow("Disparity", disp_rgb); 
    cv::imshow("V Disparity", v_disp);
    cv::imshow("U Disparity", u_disp_resized);
    cv::imshow("U Disparity Foregnd", u_disp_foregnd_resized);
    cv::imshow("Hough Vote", hough_vote_u8);
    cv::imshow("Score Image", score_u8);
    cv::waitKey(100);

  }

  return 0;
}
