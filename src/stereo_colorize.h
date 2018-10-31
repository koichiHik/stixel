
#ifndef STEREO_COLORIZE_H_
#define STEREO_COLORIZE_H_

#include <stdint.h>

namespace koichi_robotics_lib {

class StereoColorizeParams {
public:
  StereoColorizeParams() :
    grayScale(false),
    red2blue(true),
    width(1280),
    height(720),
    min(0),
    max(255){}

  bool grayScale, red2blue;
  int width, height	;
  double min, max;
};

class StereoColorize {

public:
  StereoColorize();

  ~StereoColorize();
public:

  void initialize(const StereoColorizeParams &param);

  void colorizeUInt8(uint8_t *src, uint8_t *dst8UC3);

  void colorizeFloat(float *src32FC1, uint8_t *dst8UC3);

private:
  void convertValue2RGB(const float value, uint8_t &r, uint8_t &g, uint8_t &b);

private:

  bool grayScale, red2blue;
  double width, height, bin;
  double min, max, slope, intercept;
  uint8_t *image;

};

}

#endif
