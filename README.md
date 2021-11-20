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

### Python Dependencies

In folder "python" is a conda environment file with name "environment.yml".

Importent dependencies are

- Tensorflow 1.15 (1.x should be ok, but 2.x is not supported)
- Python 3.7 (highest version which is compatible with with TF 1.15)
- open3d
- torch
- flask
- numpy
- scikit-learn
- scipy

CUDA is required to use PointNet++

## Build 

### PointNet++

To use PointNet++ you have to build the tensorflow operators under `<base>/python/pointnet2/tf_ops/`. You can use the `tf_*_compile.sh` in each folder to do so. Maybe you have to adjust the CUDA include and lib path dependend on your installation of CUDA.

### BusyBox

```
mkdir ./build
cmake -S ./src -B ./build
```
### Fast-Robust-ICP (RICP)

To use RICP you have to build the code in folder "Fast-Robust-ICP" and place the binary with name "FRICP" under "build".

## Usage

### Flask Server

``` shell
cd python
python flaskServer.py
```

### BusyBox

``` shell
./pgm -J HandcraftedStemSegmentation --in ../python/data/plant2/t1/odm_filterpoints/point_cloud.ply --Classifier 1 --NoiseFilterMinNeighbors1 30 --NoiseFilterMinNeighbors2 500 --NoiseFilterMinNeighbors3 -1 --NoiseFilterRadius1 0.8 --NoiseFilterRadius2 3.0 --SearchRadius 0.25

./pgm -J HandcraftedStemSegmentation --in ../python/data/banana/t2/shapenet/CloudWithoutBackground.ply --Classifier 1 --NoiseFilterMinNeighbors1 30 --NoiseFilterMinNeighbors2 500 --NoiseFilterMinNeighbors3 -1 --NoiseFilterRadius1 0.8 --NoiseFilterRadius2 3.0 --SearchRadius 0.025

\# 0.05 as threshold
./pgm -J HandcraftedStemSegmentation --in ~/Thesis/test/Mais/plant1/t5/Mais1T5LabeledShapenet.ply --Classifier 1 --NoiseFilterMinNeighbors1 -1 --NoiseFilterMinNeighbors2 -1 --NoiseFilterMinNeighbors3 -1 --SearchRadius 1.0  --CalculateNormals true

./pgm -J RegistrationFormat --SourceCloudPath <PATH> --TargetCloudPath <PATH> --OutputFolder <PATH>

./pgm -J Shapenet --in /home/solomon/Thesis/MasterThesis/python/data/avocado/background/odm_filterpoints/point_cloud.ply --out /home/solomon/Thesis/MasterThesis/python/data/avocado/background/shapenet/point_cloud --RemoveBackground false --MaxSubsample 1

./pgm -J RegistrationFormat --SourceCloudPath ~/Thesis/MasterThesis/data/plant2/t1.ply --TargetCloudPath ~/Thesis/MasterThesis/data/plant2/background.ply --OutputFolder ~/Thesis/python/dcp/data/Custome/plant2t1

./pgm -J RegistrationFormat --SourceCloudPath ~/Thesis/test/banana/t2/labeled/BananaT2LabeledShapenet.ply --TargetCloudPath ~/Thesis/test/banana/background_constructed/BananaBackground.ply --OutputFolder ~/Thesis/python/dcp/data/Custome/bananat2/

./pgm -J IterativeScaleRegistration --SourceCloudPath ~/Thesis/test/banana/t2/labeled/BananaT2LabeledShapenet.ply --TargetCloudPath ~/Thesis/test/banana/background_constructed/BananaBackground.ply --SubsamplePointCount 4096

./pgm -J IterativeScaleRegistration --SourceCloudPath /home/solomon/Thesis/MasterThesis/python/data/banana/t2/shapenet/point_cloudSS1BackgroundPrediction.pcd --TargetCloudPath /home/solomon/Thesis/MasterThesis/python/data/banana/background/odm_filterpoints/point_cloud.ply --SubsamplePointCount 4096

./pgm -J BackgroundRemovalPipeline --SourceCloudPath ~/Thesis/MasterThesis/python/data/avocado/background/shapenet/point_cloudSS1BackgroundPrediction.pcd --TargetCloudPath ~/Thesis/MasterThesis/python/data/avocado/background/shapenet/point_cloud.ply --OutputFolder ~/Thesis/MasterThesis/python/data/avocado/background/ --SearchRadius 0.0125

\# Using default Params
./pgm -J ManuellRegistrationPipeline --SourceCloudPath ~/Thesis/MasterThesis/python/data/avocado/t1/shapenet/CloudWithoutPlant.ply --TargetCloudPath ~/Thesis/MasterThesis/python/data/avocado/background/odm_filterpoints/point_cloud.ply

\# With all Params
./pgm -J ManuellRegistrationPipeline --SourceCloudPath ~/Thesis/MasterThesis/python/data/avocado/t1/shapenet/CloudWithoutPlant.ply --TargetCloudPath ~/Thesis/MasterThesis/python/data/avocado/background/odm_filterpoints/point_cloud.ply --NoiseFilterActive true --NoiseFilterMinNeighbors1 10 --NoiseFilterMinNeighbors2 200 --NoiseFilterRadius1 0.08 --NoiseFilterRadius2 0.2 --VoxelSize 0.015 --SegmentationAfterRegistration false
```

### Convert ply to Shapenet format
``` shell
./ConvertLabeledPlyToShapenet.sh ~/Thesis/python/data/plants/labeled/ ~/Thesis/python/data/plants/shapenet/

./ConvertLabeledPlyToShapenet.sh ~/Thesis/python/data/plants/labeled/ ~/Thesis/python/data/plants/shapenet/withRotationWithoutBackground/
```
### Plot Generation
``` shell
python reportPointNetpp.py --pathIn /home/solomon/Thesis/MasterThesis/python/pointnet2/part_seg/results/training/t13_2Classes_PartSeg_WithNorm_WithoutRot_WithoutNormals/log_train.txt --plotName T13_3C_PS_WN_NR_NN
python reportPointNetpp.py --pathIn /home/solomon/Thesis/MasterThesis/python/pointnet2/part_seg/results/training/t12_3Classes_PartSeg_WithNorm_NoRot_OnlyCenter/log_train.txt --plotName T12_3C_PS_WN_NR_OC
python reportPointNetpp.py --pathIn /home/solomon/Thesis/MasterThesis/python/pointnet2/part_seg/results/training/t11_2Classes_PartSeg_WithNorm_WithoutRot/log_train.txt --plotName T11_2C_PS_WN_NR
python reportPointNetpp.py --pathIn /home/solomon/Thesis/MasterThesis/python/pointnet2/part_seg/results/training/t10_2Classes_PartSeg_WitNorm_WithRotation/log_train.txt --plotName T10_2C_PS_WN_RR
python reportPointNetpp.py --pathIn /home/solomon/Thesis/MasterThesis/python/pointnet2/part_seg/results/training/t9_3Classes_PartSeg_WitNorm_WithRotation/log_train.txt --plotName T9_3C_PS_WN_RR
python reportPointNetpp.py --pathIn /home/solomon/Thesis/MasterThesis/python/pointnet2/part_seg/results/training/t8_2Classes_PartSeg_NoNorm_RandRot/log_train.txt --plotName T8_2C_PS_NN_RR
python reportPointNetpp.py --pathIn /home/solomon/Thesis/MasterThesis/python/pointnet2/part_seg/results/training/t7_3ClassesPartSegNoNorm/log_train.txt --plotName T7_3C_PS_NN_NR
python reportPointNetpp.py --pathIn /home/solomon/Thesis/MasterThesis/python/pointnet2/part_seg/results/training/t6_2ClassesPartSeg/log_train.txt --plotName T6_2C_PS_NN_NR
python reportPointNetpp.py --pathIn /home/solomon/Thesis/MasterThesis/python/pointnet2/part_seg/results/training/t5_3ClassesPartSeg3C/log_train.txt --plotName T5_3C_PS_WN_NR
python reportPointNetpp.py --pathIn /home/solomon/Thesis/MasterThesis/python/pointnet2/part_seg/results/training/t4_3ClassesPartSeg/log_train.txt --plotName T4_3C_PS_WN_NR
```