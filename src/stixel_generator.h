
#ifndef __STIXEL_H__
#define __STIXEL_H__

// OpenCV Library
#include <opencv2/opencv.hpp>

#include "ground_estimator.h"

class Stixel
{
public:

  Stixel() :
    left_u(0), 
    width(0), 
    vT(0), 
    vB(0), 
    disp(0.0)
  {}

  Stixel(const Stixel &obj) :
    left_u(obj.left_u),
    width(obj.width),
    vT(obj.vT),
    vB(obj.vB),
    disp(obj.disp)
  {}

  Stixel(int left_u, int width, int vT, int vB, double disp) :
    left_u(left_u),
    width(width),
    vT(vT),
    vB(vB),
    disp(disp)
  {}

  Stixel &operator=(const Stixel &obj) {
    if (this != &obj) {
      this->left_u = obj.left_u;
      this->width = obj.width;
      this->vT = obj.vT;
      this->vB = obj.vB;
      this->disp = obj.disp;
    }
    return *this;
  }

  int left_u;   // Stixel Center
  int width;    // Stixel Width
  int vT;       // Stixel Top Y
  int vB;       // Stixel Bottom Y
  double disp;  // Stixel Avg Disparity
};


class CameraParams
{
public:
  double f;
  double u0;
  double v0;
  double baseline;
  double height;
  double tilt_rad;
};

class StixelGeneratorParams
{
public:
  // Original Image
  int img_width;
  int img_height;

  // Disparity Image
  int min_disp;
  int max_disp;

  // Stixel
  int stixel_width;
  int max_depth;

  double depth_res;

  // Cost Image
  double obj_height;
  double below_under_gnd;
  double pix_thr_alpha;
  double pix_thr_intercept;

  // Dynamic Programming
  double space_smooth_fac;
  double upper_spatial_dist;

  // Height Segmentation
  double dZu;
  double upper_bnd_smooth_fac;
  double Nz;

  // Camera
  CameraParams camParam; 
};

class CoordTrans
{
public:

  void initialize(const CameraParams &param);

  double toY(double d, double v);

  double toZ(double d, double v);

  double toV(double y, double z);

  double toD(double y, double z);

private:
  CameraParams param;
  double sin_tilt, cos_tilt;
};

class StixelGenerator
{
public:

  void initialize(const StixelGeneratorParams &params);
  
  void generate_stixel(const cv::Mat &disp, const std::vector<GroundModel> &gnd_models, std::vector<Stixel> &stixels);

  void get_fs_boundary(std::vector<int> &fs_boundary_in_v, std::vector<int> &fs_boundary_in_disp, std::vector<int> &upper_bound);
  
  void get_u_disp_img(cv::Mat &u_disp_u8);

  void get_u_disp_foregnd(cv::Mat &u_disp_u8_foregnd);

  void get_score_img(cv::Mat &score_u8);

private:

  void denoise_disp_u8(const cv::Mat &disp_u8, cv::Mat &denoised_disp_u8);

  void generate_v_search_rng(int height, double obj_height, double ddisp_dv, double disp_intercept, std::vector<int> &disp_search_rng);

  void generate_disp_rng(int height, double tgt_height, double ddisp_dv, double disp_intercept, std::vector<int> &lower_disp);

  void generate_u_disp_img(const cv::Mat &disp_u8, const GroundModel &gnd_model, cv::Mat &u_disp);

  void extract_foreground_disp(const cv::Mat &u_disp_s32, const GroundModel &gnd_model, cv::Mat &u_disp_foregnd_s32);

  void convert_mat_2_u8(const cv::Mat &any_mat, cv::Mat &mat_u8);

  void calculate_fs_boundary(const cv::Mat &u_disp_foregnd_s32, const GroundModel &gnd_model, std::vector<int> &fs_boundary, std::vector<int> &fs_boundary_in_disp, std::vector<double> &fs_boundary_in_meter);

  void calculate_upper_boundary(const cv::Mat &cost_img_f32, const std::vector<double> &fs_boundary_in_meter, std::vector<int> &upper_boundary);

  void generate_membership_img(const cv::Mat &disp_u8, const std::vector<int> &fs_boundary_in_v, const std::vector<int> &fs_boundary_in_disp, cv::Mat &membership_img_f32);

  void generate_cost_img(const cv::Mat &membership_img_f32, const std::vector<int> &fs_boundary_in_v, cv::Mat &cost_img_f32);

  float calc_vote_value(int disp, int fs_disp, double fs_z);

  void calculate_stixel_height(const cv::Mat &disp_u8, const std::vector<int> &fs_boundary_in_v, const std::vector<int> &fs_boundary_in_disp, std::vector<int> &upper_boundary);

  void create_stixels(const cv::Mat &disp_u8, const std::vector<int> &fs_boundary, const std::vector<int> &upper_boundary, std::vector<Stixel> &stixels);

  void calculate_disp_v_line_base_coeff(const CameraParams &param, double &ddisp_dv, double &disp_intercept);

  void generate_stixel_elems(std::vector<Stixel> &stixels);

  void initialize_stixel_elems(std::vector<Stixel> &stixels);

private:

  CoordTrans coordTrans;

  StixelGeneratorParams param;

  cv::Mat denoised_disp_u8, u_disp_s32, u_disp_u8, depth_f32, depth_u8, score_f32, score_u8;

  cv::Mat u_disp_s32_foregnd, u_disp_u8_foregnd;

  cv::Mat dp_score_tbl_f32, dp_path_tbl_s32;

  cv::Mat member_ship_img_f32, member_ship_img_u8, cost_img_f32, cost_img_u8, integral_cost1_f32, integral_cost2_f32;

  std::vector<int> fs_boundary_in_disp, fs_boundary_in_v, disp_search_rng, lower_disp, upper_boundary;

  std::vector<double> fs_boundary_in_meter;

  std::vector<Stixel> stixels;

};




#endif
