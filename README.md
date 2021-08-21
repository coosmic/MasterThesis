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

## Usage

``` shell
./pgm -J RegistrationFormat --SourceCloudPath <PATH> --TargetCloudPath <PATH> --OutputFolder <PATH>

```

./pgm -J RegistrationFormat --SourceCloudPath ~/Thesis/MasterThesis/data/plant2/t1.ply --TargetCloudPath ~/Thesis/MasterThesis/data/plant2/background.ply --OutputFolder ~/Thesis/python/dcp/data/Custome/plant2t1

./pgm -J RegistrationFormat --SourceCloudPath ~/Thesis/test/banana/t2/labeled/BananaT2LabeledShapenet.ply --TargetCloudPath ~/Thesis/test/banana/background_constructed/BananaBackground.ply --OutputFolder ~/Thesis/python/dcp/data/Custome/bananat2/

./pgm -J IterativeScaleRegistration --SourceCloudPath ~/Thesis/test/banana/t2/labeled/BananaT2LabeledShapenet.ply --TargetCloudPath ~/Thesis/test/banana/background_constructed/BananaBackground.ply --SubsampleCount 4096


./ConvertLabeledPlyToShapenet.sh ~/Thesis/python/data/plants/labeled/ ~/Thesis/python/data/plants/shapenet/