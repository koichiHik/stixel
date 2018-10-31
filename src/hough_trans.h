
#ifndef _HOUGH_TRANS_H_
#define _HOUGH_TRANS_H_

#include "opencv2/opencv.hpp"
#include <vector>

class HoughTransParams {
public:
  int width;
  int height;
  int rho_max;
  int theta_max;
  int score_thresh;
  double rho_res;
  double theta_res;
  double vote_thr_ratio_wrt_max;
  int min_vote_per_pix;
  int max_vote_per_pix;
  double line_rho_min;
  double line_rho_max;
  double line_theta_min;
  double line_theta_max;
};

class LineElem {

public:

  LineElem() : 
    rho(0.0), theta(0.0), score(0) {}

  LineElem(const LineElem &obj) :
    rho(obj.rho), theta(obj.theta), score(obj.score){}

  LineElem(double rho, double theta, int score) :
    rho(rho), theta(theta), score(score){
  }

  LineElem &operator=(const LineElem &obj) {
    if (this != &obj) {
      this->rho = obj.rho;
      this->theta = obj.theta;
      this->score = obj.score;
    }
    return *this;
  }

  bool operator<(const LineElem &other) const {
    return score > other.score;
  }

  double rho;
  double theta;
  int score;
};

class HoughTrans {

public:
  HoughTrans();

  ~HoughTrans();

  void initialize(const HoughTransParams &params);

  void get_point_pairs(cv::Mat &gray_img, std::vector<std::pair<cv::Point2f, cv::Point2f> > &pnt_pairs);

  void get_voted_img(cv::Mat &gray_img);

  void get_line_overlaid_img(cv::Mat &rgb_img);

private:

  void get_lines(cv::Mat &gray_img, std::vector<LineElem> &lines);

  void get_top_ranked_lines(std::vector<LineElem> &lines);

  void vote_rho_theta(cv::Mat &gray_img);

  void draw_result(const cv::Mat &gray_img, std::vector<LineElem> &lines);

  void calc_pnt_in_img(double rho, double theta, cv::Point2f &p1, cv::Point2f &p2);

private:

  HoughTransParams params;
  int line_seek_u_st, line_seek_u_end, line_seek_v_st, line_seek_v_end;
  cv::Mat rho_theta, voted_img_clr, voted_img_gray, line_overlaid_img;

};

#endif // _HOUGH_TRANS_H_
