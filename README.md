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

### BusyBox

``` shell
./pgm -J RegistrationFormat --SourceCloudPath <PATH> --TargetCloudPath <PATH> --OutputFolder <PATH>

./pgm -J Shapenet --snin /home/solomon/Thesis/MasterThesis/python/data/avocado/background/odm_filterpoints/point_cloud.ply --snout /home/solomon/Thesis/MasterThesis/python/data/avocado/background/shapenet/point_cloud --RemoveBackground false --MaxSubsample 1

./pgm -J RegistrationFormat --SourceCloudPath ~/Thesis/MasterThesis/data/plant2/t1.ply --TargetCloudPath ~/Thesis/MasterThesis/data/plant2/background.ply --OutputFolder ~/Thesis/python/dcp/data/Custome/plant2t1

./pgm -J RegistrationFormat --SourceCloudPath ~/Thesis/test/banana/t2/labeled/BananaT2LabeledShapenet.ply --TargetCloudPath ~/Thesis/test/banana/background_constructed/BananaBackground.ply --OutputFolder ~/Thesis/python/dcp/data/Custome/bananat2/

./pgm -J IterativeScaleRegistration --SourceCloudPath ~/Thesis/test/banana/t2/labeled/BananaT2LabeledShapenet.ply --TargetCloudPath ~/Thesis/test/banana/background_constructed/BananaBackground.ply --SubsamplePointCount 4096

./pgm -J IterativeScaleRegistration --SourceCloudPath /home/solomon/Thesis/MasterThesis/python/data/banana/t2/shapenet/point_cloudSS1BackgroundPrediction.pcd --TargetCloudPath /home/solomon/Thesis/MasterThesis/python/data/banana/background/odm_filterpoints/point_cloud.ply --SubsamplePointCount 4096

./pgm -J BackgroundRemovalPipeline --SourceCloudPath ~/Thesis/MasterThesis/python/data/avocado/background/shapenet/point_cloudSS1BackgroundPrediction.pcd --TargetCloudPath ~/Thesis/MasterThesis/python/data/avocado/background/shapenet/point_cloud.ply --OutputFolder ~/Thesis/MasterThesis/python/data/avocado/background/ --SearchRadius 0.0125

```

### Convert ply to Shapenet format
``` shell
./ConvertLabeledPlyToShapenet.sh ~/Thesis/python/data/plants/labeled/ ~/Thesis/python/data/plants/shapenet/

./ConvertLabeledPlyToShapenet.sh ~/Thesis/python/data/plants/labeled/ ~/Thesis/python/data/plants/shapenet/withRotationWithoutBackground/
```
### Plot Generation
``` shell
python reportPointNetpp.py --pathIn /home/solomon/Thesis/MasterThesis/python/pointnet2/part_seg/results/training/t10_2Classes_PartSeg_WitNorm_WithRotation/log_train.txt --plotName T10_2C_PS_WN_RR
python reportPointNetpp.py --pathIn /home/solomon/Thesis/MasterThesis/python/pointnet2/part_seg/results/training/t9_3Classes_PartSeg_WitNorm_WithRotation/log_train.txt --plotName T9_3C_PS_WN_RR
python reportPointNetpp.py --pathIn /home/solomon/Thesis/MasterThesis/python/pointnet2/part_seg/results/training/t8_2Classes_PartSeg_NoNorm_RandRot/log_train.txt --plotName T8_2C_PS_NN_RR
python reportPointNetpp.py --pathIn /home/solomon/Thesis/MasterThesis/python/pointnet2/part_seg/results/training/t7_3ClassesPartSegNoNorm/log_train.txt --plotName T7_3C_PS_NN_NR
python reportPointNetpp.py --pathIn /home/solomon/Thesis/MasterThesis/python/pointnet2/part_seg/results/training/t6_2ClassesPartSeg/log_train.txt --plotName T6_2C_PS_NN_NR
python reportPointNetpp.py --pathIn /home/solomon/Thesis/MasterThesis/python/pointnet2/part_seg/results/training/t5_3ClassesPartSeg3C/log_train.txt --plotName T5_3C_PS_WN_NR
python reportPointNetpp.py --pathIn /home/solomon/Thesis/MasterThesis/python/pointnet2/part_seg/results/training/t4_3ClassesPartSeg/log_train.txt --plotName T4_3C_PS_WN_NR


```