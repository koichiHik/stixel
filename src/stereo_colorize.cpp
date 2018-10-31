
#include "stereo_colorize.h"
#include <algorithm>
#include <iostream>

using namespace koichi_robotics_lib;
using namespace std;

StereoColorize::StereoColorize()
  : grayScale(false),
    red2blue(true),
    width(1280),
    height(720),
    min(0),
    max(255),
    bin(1275)
{
  slope = bin / max;
  intercept = - bin / max * min;
}

StereoColorize::~StereoColorize() {

}

void StereoColorize::initialize(const StereoColorizeParams &param) {
  grayScale = param.grayScale;
  red2blue = param.red2blue;
  width = param.width;
  height = param.height;
  min = param.min;
  max = param.max;
  slope = bin / max;
  intercept = - bin / max * min;
}

void StereoColorize::convertValue2RGB(const float value, uint8_t &r, uint8_t &g, uint8_t &b) {

  int v = slope * value + intercept;

  if (0 <= v && v <= 255) {
    r = 255;
    g = v - 0;
    b = 0;
  } else if (255 < v && v <= 510) {
    r = 255 - (v - 255);
    g = 255;
    b = 0;
  } else if (510 < v && v <= 765) {
    r = 0;
    g = 255;
    b = v - 510;
  } else if (765 < v && v <= 1020) {
    r = 0;
    g = 255 - (v - 765);
    b = 255;
  } else if (1020 < v && v <= 1275) {
    r = 0;
    g = 0;
    b = 255 - (v - 1020);
  } else {
    r = 255;
    g = 255;
    b = 255;
  }

}

void StereoColorize::colorizeUInt8(uint8_t *src, uint8_t *dst8UC3) {

  uint8_t *srcPnt = src;
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      float value = *srcPnt;
      uint8_t r, g, b;
      if (red2blue) {
	convertValue2RGB(value, b, g, r);
      } else {
	convertValue2RGB(value, r, g, b);
      }

      *dst8UC3 = b;
      dst8UC3++;
      *dst8UC3 = g;
      dst8UC3++;
      *dst8UC3 = r;
      dst8UC3++;
      srcPnt++;
    }
  }
}

void StereoColorize::colorizeFloat(float *src, uint8_t *dst8UC3) {

  float *srcPnt = src;
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      float value = *srcPnt;
      uint8_t r, g, b;
      if (red2blue) {
	convertValue2RGB(value, b, g, r);
      } else {
	convertValue2RGB(value, r, g, b);
      }

      *dst8UC3 = b;
      dst8UC3++;
      *dst8UC3 = g;
      dst8UC3++;
      *dst8UC3 = r;
      dst8UC3++;
      srcPnt++;
    }
  }
}
