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

cmake -S ./src -B ./build_python_binding

## Usage

``` shell
./pgm -J RegistrationFormat --SourceCloudPath <PATH> --TargetCloudPath <PATH> --OutputFolder <PATH>

```

./pgm -J Shapenet --snin /home/solomon/Thesis/MasterThesis/python/data/avocado/background/odm_filterpoints/point_cloud.ply --snout /home/solomon/Thesis/MasterThesis/python/data/avocado/background/shapenet/point_cloud --RemoveBackground false --MaxSubsample 1

./pgm -J RegistrationFormat --SourceCloudPath ~/Thesis/MasterThesis/data/plant2/t1.ply --TargetCloudPath ~/Thesis/MasterThesis/data/plant2/background.ply --OutputFolder ~/Thesis/python/dcp/data/Custome/plant2t1

./pgm -J RegistrationFormat --SourceCloudPath ~/Thesis/test/banana/t2/labeled/BananaT2LabeledShapenet.ply --TargetCloudPath ~/Thesis/test/banana/background_constructed/BananaBackground.ply --OutputFolder ~/Thesis/python/dcp/data/Custome/bananat2/

./pgm -J IterativeScaleRegistration --SourceCloudPath ~/Thesis/test/banana/t2/labeled/BananaT2LabeledShapenet.ply --TargetCloudPath ~/Thesis/test/banana/background_constructed/BananaBackground.ply --SubsamplePointCount 4096

./pgm -J IterativeScaleRegistration --SourceCloudPath /home/solomon/Thesis/MasterThesis/python/data/banana/t2/shapenet/point_cloudSS1BackgroundPrediction.pcd --TargetCloudPath /home/solomon/Thesis/MasterThesis/python/data/banana/background/odm_filterpoints/point_cloud.ply --SubsamplePointCount 4096

./pgm -J BackgroundRemovalPipeline --SourceCloudPath ~/Thesis/MasterThesis/python/data/avocado/background/shapenet/point_cloudSS1BackgroundPrediction.pcd --TargetCloudPath ~/Thesis/MasterThesis/python/data/avocado/background/shapenet/point_cloud.ply --OutputFolder ~/Thesis/MasterThesis/python/data/avocado/background/ --SearchRadius 0.0125

./ConvertLabeledPlyToShapenet.sh ~/Thesis/python/data/plants/labeled/ ~/Thesis/python/data/plants/shapenet/