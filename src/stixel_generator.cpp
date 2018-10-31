
#include "stixel_generator.h"
#include <algorithm>

void CoordTrans::initialize(const CameraParams &param) {

  this->param = param;
  this->sin_tilt = sin(param.tilt_rad);
  this->cos_tilt = cos(param.tilt_rad);

}

double CoordTrans::toY(double d, double v) {
  return (param.baseline / d) * ((v - param.v0) * cos_tilt + param.f * sin_tilt);
}

double CoordTrans::toZ(double d, double v) {
  return (param.baseline / d) * (param.f * cos_tilt - (v - param.v0) * sin_tilt);
}

double CoordTrans::toV(double y, double z) {
  return param.v0 + param.f * (y * cos_tilt - z * sin_tilt) / (y * sin_tilt + z * cos_tilt);
}

double CoordTrans::toD(double y, double z) {
  return (param.baseline * param.f) / (y * sin_tilt + z * cos_tilt);
}

void StixelGenerator::initialize(const StixelGeneratorParams &params) {
  this->param = params;

  this->coordTrans.initialize(this->param.camParam);

  denoised_disp_u8 = cv::Mat::zeros(this->param.img_height, this->param.img_width, CV_8U);

  u_disp_u8 = cv::Mat::zeros(this->param.max_disp, this->param.img_width, CV_8U);
  u_disp_s32 = cv::Mat::zeros(this->param.max_disp, this->param.img_width, CV_32S);

  u_disp_u8_foregnd = cv::Mat::zeros(this->param.max_disp, this->param.img_width, CV_8U);
  u_disp_s32_foregnd = cv::Mat::zeros(this->param.max_disp, this->param.img_width, CV_32S);

  depth_u8 = cv::Mat::zeros(this->param.max_depth / this->param.depth_res, this->param.img_width, CV_8U);
  depth_f32 = cv::Mat::zeros(this->param.max_depth / this->param.depth_res, this->param.img_width, CV_32F);

  score_u8 = cv::Mat::zeros(this->param.img_height, this->param.img_width, CV_8U);
  score_f32 = cv::Mat::zeros(this->param.img_height, this->param.img_width, CV_32F);

  dp_path_tbl_s32 = cv::Mat::zeros(this->param.img_height, this->param.img_width, CV_32S);
  dp_score_tbl_f32 = cv::Mat::zeros(this->param.img_height, this->param.img_width, CV_32F);

  double ddisp_dv, disp_intercept;
  calculate_disp_v_line_base_coeff(this->param.camParam, ddisp_dv, disp_intercept);

  this->v_horizon = (int)std::max(-disp_intercept / ddisp_dv - this->param.dv_horizon, 0.0);

  generate_v_search_rng(this->param.img_height, this->param.obj_height, ddisp_dv, disp_intercept, this->disp_search_rng);

  generate_disp_rng(this->param.img_height, -this->param.below_under_gnd, ddisp_dv, disp_intercept, this->lower_disp);
  
}

void StixelGenerator::generate_v_search_rng(int height, double tgt_height, double ddisp_dv, double disp_intercept, std::vector<int> &disp_search_rng) {

  for (int v = 0; v < height; v++) {
    double road_disp = ddisp_dv * v + disp_intercept;
    if (this->param.min_disp <= road_disp && road_disp <= this->param.max_disp) {

      double y = coordTrans.toY(road_disp, (double)v);
      double z = coordTrans.toZ(road_disp, (double)v);
      int vb = (int)coordTrans.toV(y - tgt_height, z);

      disp_search_rng.push_back(v - vb);      
    } else {
      disp_search_rng.push_back(0);
    }
  }

}

void StixelGenerator::generate_disp_rng(int height, double tgt_height, double ddisp_dv, double disp_intercept, std::vector<int> &lower_disp) {

  for (int v = 0; v < height; v++) {
    double road_disp = ddisp_dv * v + disp_intercept;
    if (this->param.min_disp <= road_disp && road_disp <= this->param.max_disp) {

      double y = coordTrans.toY(road_disp, (double)v);
      double z = coordTrans.toZ(road_disp, (double)v);
      int d = (int)coordTrans.toD(y - tgt_height, z);
      lower_disp.push_back(d);
    } else {
      lower_disp.push_back(0);
    }
  }
  
}

void StixelGenerator::extract_foreground_disp(const cv::Mat &u_disp_s32, const GroundModel &gnd_model, cv::Mat &u_disp_foregnd_s32) {

  u_disp_foregnd_s32 = 0;

  std::cout << "Slope : " << gnd_model.slope_ddisp_dv << std::endl;
  std::cout << "Intercept : " << gnd_model.intercept_disp << std::endl;

  int32_t * const src_pnt = reinterpret_cast<int32_t *>(u_disp_s32.data);
  int32_t * const tgt_pnt = reinterpret_cast<int32_t *>(u_disp_foregnd_s32.data);

  for (int u = 0; u < u_disp_s32.cols; u++) {

    for (int v = u_disp_s32.rows - 1; 0 < v; v--) {

      int v_road = (int)(v / gnd_model.slope_ddisp_dv - gnd_model.intercept_disp);
      int pix_thr = 0.3 * this->disp_search_rng[v_road] + 5;
      int32_t ref_val = *(src_pnt + v * u_disp_s32.cols + u);

      if (pix_thr <= ref_val) {
        *(tgt_pnt + v * u_disp_s32.cols + u) = *(src_pnt + v * u_disp_s32.cols + u);
        //*(tgt_pnt + v * u_disp_s32.cols + u) = 100;
        break;
      }
    }
  }

}

void StixelGenerator::generate_stixel(const cv::Mat &disp, const std::vector<GroundModel> gnd_models, std::vector<Stixel> &stixels) {

  // Generator U Disparity Image for Free Space Calculation.
  generate_u_disp_img(disp, gnd_models[0], u_disp_s32);
  convert_mat_2_u8(u_disp_s32, u_disp_u8);
  extract_foreground_disp(u_disp_s32, gnd_models[0], u_disp_s32_foregnd);
  convert_mat_2_u8(u_disp_s32_foregnd, u_disp_u8_foregnd);

  denoise_disp_u8(disp, denoised_disp_u8);

  // Calculate Score Image.
  generate_score_img(disp, gnd_models[0], score_f32);
  //generate_score_img(denoised_disp_u8, gnd_models[0], score_f32);
  convert_mat_2_u8(score_f32, score_u8);

  // Calculate Free Space Boundary with DP.
  calculate_fs_boundary(score_f32, fs_boundary);
  
  

}

void StixelGenerator::get_fs_boundary(std::vector<int> &fs_boundary) {
  fs_boundary = this->fs_boundary;
}

void StixelGenerator::get_u_disp_img(cv::Mat &u_disp_u8) {
  this->u_disp_u8.copyTo(u_disp_u8);
}

void StixelGenerator::get_u_disp_foregnd(cv::Mat &u_disp_u8_foregnd) {
  this->u_disp_u8_foregnd.copyTo(u_disp_u8_foregnd);
}

void StixelGenerator::get_score_img(cv::Mat &score_u8) {
  this->score_u8.copyTo(score_u8);
}

void StixelGenerator::calculate_disp_v_line_base_coeff(const CameraParams &param, double &ddisp_dv, double &disp_intercept)
{
  double a = 10000000;
  double d = 0;
  double sin_t = sin(param.tilt_rad);
  double cos_t = cos(param.tilt_rad);
  double c = param.baseline / (a * param.height - d);

  ddisp_dv = c * (a * cos_t + sin_t);
  disp_intercept = c * (param.f * (a * sin_t - cos_t) - param.v0 * (a * cos_t + sin_t));
}

void StixelGenerator::generate_u_disp_img(const cv::Mat &disp_u8, const GroundModel &gnd_model, cv::Mat &u_disp) {

  u_disp = 0;
  
  unsigned char * const pnt = disp_u8.data;
  int * const u_disp_pnt = reinterpret_cast<int *>(u_disp.data);
  for (int v = 0; v < disp_u8.rows; v++) {
    unsigned char road_disp = gnd_model.slope_ddisp_dv * v + gnd_model.intercept_disp;

    if (road_disp < 0) {
      continue;
    }

    for (int u = 0; u < disp_u8.cols; u++) {
      unsigned char disp = *(pnt + v * disp_u8.cols + u);
      if (disp < param.min_disp || param.max_disp < disp || disp < road_disp + 2) {
        continue;
      }

      int *tmp = u_disp_pnt + disp * u_disp.cols + u;
      *tmp = *tmp + 1;
    }
  }

}

void StixelGenerator::denoise_disp_u8(const cv::Mat &disp_u8, cv::Mat &denoised_disp_u8) {

  denoised_disp_u8 = 0;

  unsigned char * const pnt = disp_u8.data;
  unsigned char * const tgt_pnt = denoised_disp_u8.data;

  for (int v=0; v<disp_u8.rows; v++) {
    for (int u=0; u<disp_u8.cols; u++) {

      unsigned char disp = *(pnt + v * disp_u8.cols + u);
      // If disparity is in range, apply lower bound.
      if (param.min_disp <= disp  || disp <= param.max_disp) {
        *(tgt_pnt + v * disp_u8.cols + u) = std::max(disp, static_cast<unsigned char>(lower_disp[v]));
      } else {
        *(tgt_pnt + v * disp_u8.cols + u) = disp;
      }
    }
  }
}


void StixelGenerator::convert_mat_2_u8(const cv::Mat &any_mat, cv::Mat &mat_u8) {

  double min, max;
  cv::minMaxLoc(any_mat, &min, &max);
  any_mat.convertTo(mat_u8, CV_8U, 255 / (max - min), -255 * min / (max - min));

}

void StixelGenerator::calculate_fs_boundary(const cv::Mat &cost_f32, std::vector<int> &fs_boundary) {
  dp_score_tbl_f32 = 0.0;
  dp_path_tbl_s32 = 0;

  float * const score_pnt = reinterpret_cast<float *>(cost_f32.data);
  int * const dp_path_tbl_pnt = reinterpret_cast<int *>(dp_path_tbl_s32.data);
  float * const dp_score_tbl_pnt = reinterpret_cast<float *>(dp_score_tbl_f32.data);

  int d_row = 20;

  // Create DP table.
  for (int u = 1; u < cost_f32.cols; u++) {

    for (int v = this->v_horizon; v < cost_f32.rows - d_row; v++) {
      const float * ref_pnt = score_pnt + v * cost_f32.cols + u;
      int * tgt_path_pnt = dp_path_tbl_pnt + v * cost_f32.cols + u;
      float * tgt_score_pnt = dp_score_tbl_pnt + v * cost_f32.cols + u;

      int min_v = 0;
      float min_score = std::numeric_limits<float>::max();
      for (int v_in_last_col = this->v_horizon; v_in_last_col < cost_f32.rows - d_row; v_in_last_col++) {

        float self_cost = *(dp_score_tbl_pnt + v_in_last_col * cost_f32.cols + (u - 1));
        self_cost = self_cost + std::abs(cost_f32.rows - v_in_last_col) * 3.0;
        double jump_dist = std::min(static_cast<double>(std::abs(v - v_in_last_col)), this->param.upper_spatial_dist);
        float cost = self_cost + this->param.space_smooth_fac * jump_dist;

        if (cost < min_score) {
          min_score = cost;
          min_v = v_in_last_col;
        }
       
      }
      *(tgt_path_pnt) = min_v;
      *(tgt_score_pnt) = *(ref_pnt) + min_score;
    }
  }

  // Extract Path.
  int min_v = 0;
  float min_score = std::numeric_limits<float>::max();
  for (int v = this->v_horizon; v < cost_f32.rows - d_row; v++) {
    float score = *(dp_score_tbl_pnt + v * cost_f32.cols + cost_f32.cols - 1);
    if (score < min_score) {
      min_v = v;
    }
  }
  
  fs_boundary.push_back(min_v);
  for (int u = cost_f32.cols - 2; 0 <= u; u--) {
    int v = *(dp_path_tbl_pnt + min_v * cost_f32.cols + u);
    min_v = v;
    fs_boundary.push_back(min_v);
  }

  std::reverse(fs_boundary.begin(), fs_boundary.end());
}

void StixelGenerator::calculate_stixel_height(const cv::Mat &cost_f32, const std::vector<int> &fs_boundary, std::vector<int> &upper_boundary) {

  


}

void StixelGenerator::create_stixels(const cv::Mat &disp_u8, const std::vector<int> &fs_boundary, const std::vector<int> &upper_boundary, std::vector<Stixel> stixels) {

  



}

void StixelGenerator::generate_score_img(const cv::Mat &disp_u8, const GroundModel &gnd_model, cv::Mat &score_f32) {

  score_f32 = 255.0;

  unsigned char * const disp_pnt = disp_u8.data;
  float * const score_pnt = reinterpret_cast<float *>(score_f32.data);

  for (int v = this->param.ignr_drow_low; v < disp_u8.rows - this->param.ignr_drow_up; v++) {

    float road_disp = gnd_model.slope_ddisp_dv * v + gnd_model.intercept_disp;

    for (int u = 0; u < disp_u8.cols; u++) {

      float disp = static_cast<float>(*(disp_pnt + v * disp_u8.cols + u));
      
      if (param.min_disp <= disp && disp <= param.max_disp) {
        float road_score = std::abs(road_disp - disp);
        float obst_score = 0;

        int search_rng = disp_search_rng[v];
        for (int dd = -search_rng; dd <= 0; dd++) {
          float tmp = *(disp_pnt + (v + dd) * disp_u8.cols + u);
          
          if (param.min_disp <= tmp && tmp <=param.max_disp) {
            //obst_score = obst_score + std::abs(road_disp - tmp) / (float)std::max(search_rng, 1);
            obst_score = obst_score + std::abs(road_disp - tmp);
          }
        }

        *(score_pnt + v * disp_u8.cols + u) = this->param.road_dev_cost * road_score + this->param.obst_dev_cost * obst_score;
      }
    }
  }
}






