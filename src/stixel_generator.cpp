
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

  dp_path_tbl_s32 = cv::Mat::zeros(this->param.max_depth / this->param.depth_res, this->param.img_width, CV_32S);
  dp_score_tbl_f32 = cv::Mat::zeros(this->param.max_depth / this->param.depth_res, this->param.img_width, CV_32F);

  score_u8 = cv::Mat::zeros(this->param.img_height, this->param.img_width, CV_8U);
  score_f32 = cv::Mat::zeros(this->param.img_height, this->param.img_width, CV_32F);

  member_ship_img_u8 = cv::Mat::zeros(this->param.img_height, this->param.img_width, CV_8U);
  member_ship_img_f32 = cv::Mat::zeros(this->param.img_height, this->param.img_width, CV_32F);

  cost_img_u8 = cv::Mat::zeros(this->param.img_height, this->param.img_width, CV_8U);
  cost_img_f32 = cv::Mat::zeros(this->param.img_height, this->param.img_width, CV_32F);
  integral_cost1_f32 = cv::Mat::zeros(this->param.img_height, this->param.img_width, CV_32F);
  integral_cost2_f32 = cv::Mat::zeros(this->param.img_height, this->param.img_width, CV_32F);

  double ddisp_dv, disp_intercept;
  calculate_disp_v_line_base_coeff(this->param.camParam, ddisp_dv, disp_intercept);

  generate_v_search_rng(this->param.img_height, this->param.obj_height, ddisp_dv, disp_intercept, this->disp_search_rng);

  generate_disp_rng(this->param.img_height, -this->param.below_under_gnd, ddisp_dv, disp_intercept, this->lower_disp);

  generate_stixel_elems(this->stixels);
  
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

  int32_t * const src_pnt = reinterpret_cast<int32_t *>(u_disp_s32.data);
  int32_t * const tgt_pnt = reinterpret_cast<int32_t *>(u_disp_foregnd_s32.data);

  for (int u = 0; u < u_disp_s32.cols; u++) {

    for (int v = u_disp_s32.rows - 1; 0 < v; v--) {

      int v_road = (int)(v / gnd_model.slope_ddisp_dv - gnd_model.intercept_disp);
      int pix_thr = this->param.pix_thr_alpha * this->disp_search_rng[v_road] + this->param.pix_thr_intercept;
      int32_t ref_val = *(src_pnt + v * u_disp_s32.cols + u);

      if (pix_thr <= ref_val) {
        //*(tgt_pnt + v * u_disp_s32.cols + u) = *(src_pnt + v * u_disp_s32.cols + u);
        *(tgt_pnt + v * u_disp_s32.cols + u) = 100;
        break;
      }
    }
  }

}

void StixelGenerator::generate_stixel(const cv::Mat &disp, const std::vector<GroundModel> &gnd_models, std::vector<Stixel> &stixels) {

  // Generator U Disparity Image for Free Space Calculation.
  generate_u_disp_img(disp, gnd_models[0], u_disp_s32);
  convert_mat_2_u8(u_disp_s32, u_disp_u8);
  extract_foreground_disp(u_disp_s32, gnd_models[0], u_disp_s32_foregnd);
  convert_mat_2_u8(u_disp_s32_foregnd, u_disp_u8_foregnd);

  denoise_disp_u8(disp, denoised_disp_u8);

  // Calculate Free Space Boundary with DP.
  calculate_fs_boundary(u_disp_s32_foregnd, gnd_models[0], fs_boundary_in_v, fs_boundary_in_disp, fs_boundary_in_meter);

  // Height calculation.
  calculate_stixel_height(disp, fs_boundary_in_v, fs_boundary_in_disp, upper_boundary);

  create_stixels(disp, fs_boundary_in_v, upper_boundary, this->stixels);

  stixels = this->stixels;
}

void StixelGenerator::get_fs_boundary(std::vector<int> &fs_boundary_in_v, std::vector<int> &fs_boundary_in_disp, std::vector<int> &upper_bound) {
  fs_boundary_in_v = this->fs_boundary_in_v;
  fs_boundary_in_disp = this->fs_boundary_in_disp;
  upper_bound = this->upper_boundary;
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

void StixelGenerator::calculate_fs_boundary(const cv::Mat &u_disp_foregnd_s32, const GroundModel &gnd_model, std::vector<int> &fs_boundary_in_v, std::vector<int> &fs_boundary_in_disp, std::vector<double> &fs_boundary_in_meter) {
  dp_score_tbl_f32 = 0.0;
  dp_path_tbl_s32 = 0;

  int32_t * const udisp_pnt = reinterpret_cast<int32_t *>(u_disp_foregnd_s32.data);
  int32_t * const dp_path_tbl_pnt = reinterpret_cast<int32_t *>(dp_path_tbl_s32.data);
  float * const dp_score_tbl_pnt = reinterpret_cast<float *>(dp_score_tbl_f32.data);

  // Create DP table.
  for (int u = this->param.max_disp; u < u_disp_foregnd_s32.cols; u++) {

    for (int disp = 0; disp < u_disp_foregnd_s32.rows; disp++) {
      int32_t * const ref_pnt = udisp_pnt + disp * u_disp_foregnd_s32.cols + u;
      int32_t * tgt_path_pnt = dp_path_tbl_pnt + disp * u_disp_foregnd_s32.cols + u;
      float * tgt_score_pnt = dp_score_tbl_pnt + disp * u_disp_foregnd_s32.cols + u;

      int32_t max_disp = 0; 
      float max_score = std::numeric_limits<float>::min();

      int v1 = (int)((disp - gnd_model.intercept_disp) / gnd_model.slope_ddisp_dv);
      double z1 = this->coordTrans.toZ(disp, v1);

      for (int disp_in_last_col = 0; disp_in_last_col < u_disp_foregnd_s32.rows; disp_in_last_col++) {

        int v2 = (int)((disp_in_last_col - gnd_model.intercept_disp) / gnd_model.slope_ddisp_dv);
        double z2 = this->coordTrans.toZ(disp_in_last_col, v2);

        float last_elem_score = *(dp_score_tbl_pnt + disp_in_last_col * u_disp_foregnd_s32.cols + (u - 1));
        double jump_penalty = std::min(static_cast<double>(std::abs(z1 - z2)), this->param.upper_spatial_dist);
        float connection_score = last_elem_score - this->param.space_smooth_fac * jump_penalty;

        if (max_score < connection_score) {
          max_score = connection_score;
          max_disp = disp_in_last_col;
        }
       
      }
      *(tgt_path_pnt) = max_disp;
      *(tgt_score_pnt) = *(ref_pnt) + max_score;
    }
  }

  // Extract Path.
  int max_disp = 0;
  float max_score = std::numeric_limits<float>::min();
  for (int disp = 0; disp < u_disp_foregnd_s32.rows; disp++) {
    float score = *(dp_score_tbl_pnt + disp * u_disp_foregnd_s32.cols + u_disp_foregnd_s32.cols - 1);
    if (max_score < score) {
      max_disp = disp;
    }
  }
  
  double max_v_dbl = (max_disp - gnd_model.intercept_disp) / gnd_model.slope_ddisp_dv; 
  int max_v = (int)(max_v_dbl);
  fs_boundary_in_disp.push_back(max_disp);
  fs_boundary_in_v.push_back(max_v);
  fs_boundary_in_meter.push_back(coordTrans.toZ(max_disp, max_v_dbl));

  for (int u = u_disp_foregnd_s32.cols - 2; this->param.max_disp < u; u--) {
    int disp = *(dp_path_tbl_pnt + max_disp * u_disp_foregnd_s32.cols + u);
    max_disp = disp;
    max_v_dbl = (max_disp - gnd_model.intercept_disp) / gnd_model.slope_ddisp_dv;
    max_v = (int)(max_v_dbl);
    fs_boundary_in_disp.push_back(max_disp);
    fs_boundary_in_v.push_back(max_v);
    fs_boundary_in_meter.push_back(coordTrans.toZ(max_disp, max_v_dbl));
    
  }

  for (int u = this->param.max_disp; 0 <= u; u--) {
    //fs_boundary_in_disp.push_back(max_disp);
    //fs_boundary_in_v.push_back(max_v);
    //fs_boundary_in_meter.push_back(max_v_dbl);
    fs_boundary_in_disp.push_back(this->param.max_disp);
    fs_boundary_in_v.push_back(332);
    fs_boundary_in_meter.push_back(0);
  }

  std::reverse(fs_boundary_in_disp.begin(), fs_boundary_in_disp.end());
  std::reverse(fs_boundary_in_v.begin(), fs_boundary_in_v.end());
  std::reverse(fs_boundary_in_meter.begin(), fs_boundary_in_meter.end());
}

void StixelGenerator::generate_membership_img(const cv::Mat &disp_u8, const std::vector<int> &fs_boundary_in_v, const std::vector<int> &fs_boundary_in_disp, cv::Mat &member_ship_img_f32) {

  member_ship_img_f32 = 0;

  unsigned char * const ref_pnt = disp_u8.data;
  float * const tgt_pnt = reinterpret_cast<float *>(member_ship_img_f32.data);
  
  for (int u = this->param.max_disp; u < disp_u8.cols; u++) {

    int fs_v = fs_boundary_in_v[u];
    int fs_disp = fs_boundary_in_disp[u];
    double fs_z = coordTrans.toZ(fs_disp, fs_v);

    for (int v = fs_v - 1; 0 <= v; v--) {

      if (v < 0 || disp_u8.rows <= v) {        
        continue;
      }

      int disp = static_cast<int>(*(ref_pnt + v * disp_u8.cols + u));
      *(tgt_pnt + v * disp_u8.cols + u) = -calc_vote_value(disp, fs_disp, fs_z);

    }

  }
 
}

void StixelGenerator::generate_cost_img(const cv::Mat &membership_img_f32, const std::vector<int> &fs_boundary_in_v, cv::Mat &cost_img_f32) {

  cost_img_f32 = 0;
  integral_cost1_f32 = 0;
  integral_cost2_f32 = 0;

  float * const mem_pnt = reinterpret_cast<float *>(membership_img_f32.data);
  float * const integ_pnt1 = reinterpret_cast<float *>(integral_cost1_f32.data);
  float * const integ_pnt2 = reinterpret_cast<float *>(integral_cost2_f32.data);
  
  for (int u = 0; u < membership_img_f32.cols; u++) {
    for (int v = 1; v < std::min(fs_boundary_in_v[u], membership_img_f32.rows); v++) {
      float const * ref_pnt = mem_pnt + v * membership_img_f32.cols + u;
      float const * tgt_pnt_last_row = integ_pnt1 + (v - 1) * membership_img_f32.cols + u;
      float * tgt_pnt = integ_pnt1 + v * membership_img_f32.cols + u;
      *tgt_pnt = *ref_pnt + *tgt_pnt_last_row;
    }
  }

  for (int u = 0; u < membership_img_f32.cols; u++) {
    for (int v = std::min(fs_boundary_in_v[u], membership_img_f32.rows); 0 <= v; v--) {
      float const * ref_pnt = mem_pnt + v * membership_img_f32.cols + u;
      float const * tgt_pnt_last_row = integ_pnt2 + (v + 1) * membership_img_f32.cols + u;
      float * tgt_pnt = integ_pnt2 + v * membership_img_f32.cols + u;
      *tgt_pnt = *ref_pnt + *tgt_pnt_last_row;
    }
  }

  cost_img_f32 = integral_cost1_f32 - integral_cost2_f32;
  double min, max;
  cv::minMaxLoc(integral_cost2_f32, &min, &max);
  convert_mat_2_u8(cost_img_f32, cost_img_u8);
  cv::imshow("Cost Img", cost_img_u8);

}

float StixelGenerator::calc_vote_value(int disp, int fs_disp, double fs_z) {

  double dDu = std::abs(fs_disp - param.camParam.baseline * param.camParam.f / (fs_z + param.dZu));
  double ddisp = std::abs(disp - fs_disp);
  double pow_elem = 1-std::pow((ddisp/dDu), 2.0);
  return std::pow(2, pow_elem) - 1;

}

void StixelGenerator::calculate_stixel_height(const cv::Mat &disp_u8, const std::vector<int> &fs_boundary_in_v, const std::vector<int> &fs_boundary_in_disp, std::vector<int> &upper_boundary) {

  generate_membership_img(disp_u8, fs_boundary_in_v, fs_boundary_in_disp, member_ship_img_f32);

  convert_mat_2_u8(member_ship_img_f32, member_ship_img_u8);

  generate_cost_img(member_ship_img_f32, fs_boundary_in_v, cost_img_f32);

  calculate_upper_boundary(cost_img_f32, fs_boundary_in_meter, upper_boundary);

}


void StixelGenerator::calculate_upper_boundary(const cv::Mat &cost_img_f32, const std::vector<double> &fs_boundary_in_meter, std::vector<int> &upper_boundary) {

  dp_score_tbl_f32 = 0;
  dp_path_tbl_s32 = 0;

  float * const score_img_pnt = reinterpret_cast<float *>(cost_img_f32.data);
  float * const dp_score_tbl_pnt = reinterpret_cast<float *>(dp_score_tbl_f32.data);
  int32_t * const dp_path_tbl_pnt = reinterpret_cast<int32_t *>(dp_path_tbl_s32.data);
  
  for (int u = this->param.max_disp; u < cost_img_f32.cols; u++) {

    for (int v = 0; v < cost_img_f32.rows; v++) {

      float * const ref_pnt = score_img_pnt + v * cost_img_f32.cols + u;
      float * tgt_score_pnt = dp_score_tbl_pnt + v * cost_img_f32.cols + u;
      int32_t * tgt_path_pnt = dp_path_tbl_pnt + v * cost_img_f32.cols + u;

      int32_t max_v = 0;
      float max_score = std::numeric_limits<float>::min();

      for (int v_in_last_col = 0; v_in_last_col < cost_img_f32.rows; v_in_last_col++) {
        
        float last_elem_score = *(dp_score_tbl_pnt + v_in_last_col * cost_img_f32.cols + (u - 1));

        double fs_z_diff = std::abs(fs_boundary_in_meter[u+1] - fs_boundary_in_meter[u]);
        double jump_penalty = std::abs(v - v_in_last_col) * std::max(0.0, 1 - fs_z_diff / this->param.Nz);
        float connection_score = last_elem_score - this->param.upper_bnd_smooth_fac * jump_penalty;

        if (max_score < connection_score) {
          max_v = v_in_last_col;
          max_score = connection_score;
        }

      }
      *(tgt_path_pnt) = max_v;
      *(tgt_score_pnt) = *(ref_pnt) + max_score;
    }
  }

  // Extract Path
  int32_t max_v = 0;
  float max_score = std::numeric_limits<float>::min();
  for (int v = 0; v < dp_score_tbl_f32.rows; v++) {
    float score = *(dp_score_tbl_pnt + v * dp_score_tbl_f32.cols + dp_score_tbl_f32.cols - 1);
    if (max_score < score) {
      max_v = v;
    }
  }
  
  upper_boundary.push_back(max_v);
  for (int u = dp_score_tbl_f32.cols - 2; this->param.max_disp < u; u--) {
    int32_t v = *(dp_path_tbl_pnt + max_v * dp_score_tbl_f32.cols + u);
    max_v = v;
    upper_boundary.push_back(max_v);
  }

  for (int u = this->param.max_disp; 0 <= u; u--) {
    upper_boundary.push_back(dp_path_tbl_s32.rows - 1);    
    //upper_boundary.push_back(max_v);
  }

  std::reverse(upper_boundary.begin(), upper_boundary.end());

}

void StixelGenerator::create_stixels(const cv::Mat &disp_u8, const std::vector<int> &fs_boundary, const std::vector<int> &upper_boundary, std::vector<Stixel> &stixels) {

  initialize_stixel_elems(stixels);

  float disp_sum_in_stxl = 0.0;
  unsigned char * const ref_disp_pnt = reinterpret_cast<unsigned char *>(disp_u8.data);
 
  for (auto &stxl : this->stixels) {

    if (stxl.left_u + stxl.width <= this->param.max_disp) {
      continue;
    }

    int disp_cnt = 0, v_cnt = 0;
    double disp_sum = 0.0, vT_sum = 0.0, vB_sum = 0.0;

    // Sum up all disparity contained in stixel.
    for (int d_width = 0; d_width < stxl.width; d_width++) {
      int tgt_u = stxl.left_u + d_width;
      v_cnt++;
      vB_sum += fs_boundary[tgt_u];
      vT_sum += upper_boundary[tgt_u];
      for (int tgt_v = fs_boundary[tgt_u]; upper_boundary[tgt_u] <= tgt_v; tgt_v--) {
        disp_cnt++;
        disp_sum += *(ref_disp_pnt + disp_u8.cols * tgt_v + tgt_u);
      }
    }  

    stxl.disp = disp_sum / disp_cnt;
    stxl.vT = static_cast<int32_t>(vT_sum / v_cnt + 0.5);
    stxl.vB = static_cast<int32_t>(vB_sum / v_cnt + 0.5);

    disp_cnt = 0;
    v_cnt = 0;
    v_cnt = 0;
    disp_sum = 0;
    vT_sum = 0;
    vB_sum = 0;
  }

}

void StixelGenerator::generate_stixel_elems(std::vector<Stixel> &stixels) {

  int32_t stxl_no = static_cast<int32_t>(this->param.img_width / this->param.stixel_width);
  if (this->param.img_width % this->param.stixel_width != 0) {
    stxl_no++;
  }

  for (int32_t idx = 0; idx < stxl_no; idx++) {
    Stixel elem;
    elem.left_u = idx * this->param.stixel_width;
    elem.width = std::min(this->param.stixel_width, this->param.img_width - elem.left_u);
    elem.vT = 0;
    elem.vB = 0;
    elem.disp = 0.0;
    stixels.push_back(elem);
  }

}

void StixelGenerator::initialize_stixel_elems(std::vector<Stixel> &stixels) {

  for (auto stixel : stixels) {
    stixel.vB = 0;
    stixel.vT = 0;
    stixel.disp = 0;
  }

}

