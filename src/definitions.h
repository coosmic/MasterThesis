#pragma once
#include <pcl/point_types.h>
#include <pcl/io/io.h>

#define DEBUG true

#define PointTypePCL pcl::PointXYZRGBNormal

#define PointTypeRegistration pcl::PointXYZRGB

#define Cloud pcl::PointCloud<PointTypePCL>

enum ColorChannel{
  r,g,b
};

#define BackgroundLabel 2
#define StemLabel 1
#define LeaveLabel 0