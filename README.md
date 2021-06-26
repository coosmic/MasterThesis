# Readme

## Pre-Requirements

Add following to `sift_keypoint.h` in you pcl installation:

``` cpp
  template<>
  struct SIFTKeypointFieldSelector<PointXYZRGBNormal>
  {
    inline float
    operator () (const PointXYZRGBNormal & p) const
    {
      return p.curvature;
    }
  };
```