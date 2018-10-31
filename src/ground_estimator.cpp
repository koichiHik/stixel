
#include "ground_estimator.h"

GroundEstimator::GroundEstimator() {

}

GroundEstimator::~GroundEstimator() {

}

void GroundEstimator::initialize(GroundEstimatorParams &param) {
  this->param = param;

  v_disp_u8 = cv::Mat::zeros(this->param.v_disp_height, this->param.v_disp_width, CV_8U);
  v_disp_s32 = cv::Mat::zeros(this->param.v_disp_height, this->param.v_disp_width, CV_32S);

  hough_trans.initialize(param.hough_param);
}

void GroundEstimator::estimate_ground(cv::Mat &disp_u8, std::vector<GroundModel> &models) {

  generate_v_disp_img(disp_u8, v_disp_s32);
  convert_v_disp_s32_2_u8(v_disp_s32, v_disp_u8);

  std::vector<std::pair<cv::Point2f, cv::Point2f> > pnt_pairs;
  hough_trans.get_point_pairs(v_disp_u8, pnt_pairs);

  models.clear();
  GroundModel model;
  convert_pnt_pairs_2_GroundModel(pnt_pairs[0], model);
  models.push_back(model);

}

void GroundEstimator::get_v_disp_img(cv::Mat &v_disp_u8) {
  this->v_disp_u8.copyTo(v_disp_u8);
}

void GroundEstimator::get_hough_vote_img(cv::Mat &hough_vote_u8) {
  hough_trans.get_voted_img(hough_vote_u8);
}

void GroundEstimator::get_line_overlaid_img(cv::Mat &rgb_img) {
  hough_trans.get_line_overlaid_img(rgb_img);
}

void GroundEstimator::convert_v_disp_s32_2_u8(cv::Mat &v_disp_s32, cv::Mat &v_disp_u8) {
  double min, max;
  cv::minMaxLoc(v_disp_s32, &min, &max);
  v_disp_s32.convertTo(v_disp_u8, CV_8U, 255 / (max - min), 255 * min / (max - min));
}

void GroundEstimator::generate_v_disp_img(cv::Mat &disp_u8, cv::Mat &v_disp) {

  v_disp = 0;
  
  unsigned char * const pnt = disp_u8.data;
  int * const v_disp_pnt = reinterpret_cast<int *>(v_disp.data);
  for (int v = 0; v < disp_u8.rows; v++) {
    for (int u = 0; u < disp_u8.cols; u++) {
      unsigned char disp = *(pnt + v * disp_u8.cols + u);
      if (disp < param.min_disp || param.max_disp < disp) {
        continue;
      }

      int *tmp = v_disp_pnt + v * v_disp.cols + disp;
      *tmp = *tmp + 1;
    }
  }

}

void GroundEstimator::convert_pnt_pairs_2_GroundModel(const std::pair<cv::Point2f, cv::Point2f> &pnt_pair, GroundModel &model) {

  if (std::abs(pnt_pair.first.y - pnt_pair.second.y) < 0.000001f) {
    std::cout << "Ground slope is INF!!" << std::endl;
  } else {
    model.slope_ddisp_dv = (pnt_pair.first.x - pnt_pair.second.x) / (pnt_pair.first.y - pnt_pair.second.y);
    model.intercept_disp = pnt_pair.first.x - pnt_pair.first.y * (pnt_pair.first.x - pnt_pair.second.x) / (pnt_pair.first.y - pnt_pair.second.y);
  }

}
