
#include <iostream>
#include <string>
#include <cmath>
#include <set>

#include "opencv2/opencv.hpp"
#include "hough_trans.h"

HoughTrans::HoughTrans(){

}

HoughTrans::~HoughTrans() {

}

void HoughTrans::initialize(const HoughTransParams &params) {

  this->params = params;
  rho_theta = cv::Mat(this->params.rho_max / this->params.rho_res, this->params.theta_max / this->params.theta_res, CV_32S);
  voted_img_clr = cv::Mat(this->params.rho_max / this->params.rho_res, this->params.theta_max / this->params.theta_res, CV_8UC1);
  voted_img_clr = cv::Mat(this->params.rho_max / this->params.rho_res, this->params.theta_max / this->params.theta_res, CV_8UC3);
  line_overlaid_img = cv::Mat(this->params.height, this->params.width, CV_8UC3);

  line_seek_v_st = std::max(0, static_cast<int>(params.line_rho_min));
  line_seek_v_end = std::min(static_cast<int>(params.line_rho_max), rho_theta.rows);
  line_seek_u_st = std::max(0, static_cast<int>(params.line_theta_min));
  line_seek_u_end = std::min(static_cast<int>(params.line_theta_max), rho_theta.cols);

}

void HoughTrans::get_top_ranked_lines(std::vector<LineElem> &lines) {

  lines.clear();

  int * const vote_st_pnt = reinterpret_cast<int *>(rho_theta.data);

  for (int v = line_seek_v_st; v < line_seek_v_end; v++) {
    for (int u = line_seek_u_st; u < line_seek_u_end; u++) {

      double rho = v * params.rho_res;
      double theta = u * params.theta_res;
      int score = *(vote_st_pnt + v * rho_theta.cols + u);

      if (score <= params.score_thresh) {
        continue;
      }
      LineElem line2(rho, theta, score);
      LineElem line = line2;
      lines.push_back(line);
    }
  }

  std::sort(lines.begin(), lines.end());  

}

void HoughTrans::get_voted_img(cv::Mat &gray_img) {
  double min, max;
  cv::minMaxLoc(rho_theta, &min, &max);
  rho_theta.convertTo(voted_img_gray, CV_8U, 255.0 / (max - min), - (255.0 * min) / (max - min));
  cv::cvtColor(voted_img_gray, voted_img_clr, CV_GRAY2RGB);

  cv::rectangle(voted_img_clr, cv::Point(line_seek_u_st, line_seek_v_st), cv::Point(line_seek_u_end, line_seek_v_end), cv::Scalar(0, 0, 200), 1, 4);
  gray_img = voted_img_clr;
}

void HoughTrans::get_line_overlaid_img(cv::Mat &rgb_img) {
  rgb_img = line_overlaid_img;
}

void HoughTrans::vote_rho_theta(cv::Mat &gray_img) {

  // Initialize Buffer.
  rho_theta = 0;

  unsigned char * const img_pnt = gray_img.data;
  int * const vote_st_pnt = reinterpret_cast<int *>(rho_theta.data);

  double min, max;
  cv::minMaxLoc(gray_img, &min, &max);

  // Outer loop for image 
  for (int v = 0; v < gray_img.rows; v++) {
    for (int u = 0; u < gray_img.cols; u++) {

      unsigned char val = *(img_pnt + v * gray_img.cols + u);

      if (val >= params.vote_thr_ratio_wrt_max * max) {

        // Inner loop for voting
        for (int theta_col = 0;  theta_col < rho_theta.cols; theta_col++) {

          // Rho-Theta Calculation
          double theta = params.theta_res * theta_col * M_PI / 180.0;
          double rho_abs = std::abs(u * cos(theta) + v * sin(theta));

          if (rho_abs <= params.rho_max) {
            // Hough Vote
            int rho_row = std::abs(static_cast<int>(rho_abs / params.rho_res));
            int *tgt_pnt = vote_st_pnt + rho_row * rho_theta.cols + theta_col;
            *tgt_pnt = *tgt_pnt + std::min(std::max((int)val, params.min_vote_per_pix), params.max_vote_per_pix);
          }

        }
      }
    }
  }
}

void HoughTrans::get_lines(cv::Mat &gray_img, std::vector<LineElem> &lines) {

  // 1. Vote
  vote_rho_theta(gray_img);

  // 2. Get Highly Ranked Lines
  get_top_ranked_lines(lines);

  // 3. Draw result image
  draw_result(gray_img, lines);

}

void HoughTrans::get_point_pairs(cv::Mat &gray_img, std::vector<std::pair<cv::Point2f, cv::Point2f> > &pnt_pairs) {
  std::vector<LineElem> line_elems;
  get_lines(gray_img, line_elems);

  for (auto line_elem : line_elems) {
    cv::Point2f pt1, pt2;
    calc_pnt_in_img(line_elem.rho, line_elem.theta, pt1, pt2);
    std::pair<cv::Point2f, cv::Point2f> pair = std::make_pair(pt1, pt2);
    pnt_pairs.push_back(pair);
  }
}

void HoughTrans::draw_result(const cv::Mat &gray_img, std::vector<LineElem> &lines) {

  cv::cvtColor(gray_img, line_overlaid_img, CV_GRAY2RGB);

  for (int i = 0; i < 1; i++) {
    cv::Point2f pt1, pt2;
    calc_pnt_in_img(lines[i].rho, lines[i].theta, pt1, pt2);
    cv::line(line_overlaid_img, pt1, pt2, cv::Scalar(0, 0, 255), 1);
  }

}

void HoughTrans::calc_pnt_in_img(double rho, double theta, cv::Point2f &p1, cv::Point2f &p2) {

  int x1, y1, x2, y2;
  double theta_rad = theta * M_PI / 180.0;
  if ((theta < 45.0) || (135.0 < theta)) {
    y1 = -10000;
    y2 = 10000;
    x1 = -(sin(theta_rad)/cos(theta_rad)) * y1 + rho / cos(theta_rad);
    x2 = -(sin(theta_rad)/cos(theta_rad)) * y2 + rho / cos(theta_rad);
  } else {
    x1 = -10000;
    x2 = 10000;
    y1 = -(cos(theta_rad)/sin(theta_rad)) * x1 + rho / sin(theta_rad);
    y2 = -(cos(theta_rad)/sin(theta_rad)) * x2 + rho / sin(theta_rad);
  }
  p1.x = x1;
  p1.y = y1;
  p2.x = x2;
  p2.y = y2;
}


