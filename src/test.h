#include <iostream>

#include "definitions.h"

#include <pcl/io/pcd_io.h>
#include <pcl/keypoints/sift_keypoint.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

int
testSIFTKeypoints(pcl::PointCloud<PointTypePCL>::Ptr cloudIn)
{
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  copyPointCloud(*cloudIn, *cloud);
  
  // Parameters for sift computation
  const float min_scale = 0.1f;
  const int n_octaves = 6;
  const int n_scales_per_octave = 10;
  const float min_contrast = 0.5f;
  
  
  // Estimate the sift interest points using Intensity values from RGB values
  pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointWithScale> sift;
  pcl::PointCloud<pcl::PointWithScale> result;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB> ());
  sift.setSearchMethod(tree);
  sift.setScales(min_scale, n_octaves, n_scales_per_octave);
  sift.setMinimumContrast(min_contrast);
  sift.setInputCloud(cloud);
  sift.compute(result);
  
  // Copying the pointwithscale to pointxyz so as visualize the cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp (new pcl::PointCloud<pcl::PointXYZ>);
  copyPointCloud(result, *cloud_temp);

  // Saving the resultant cloud 
  std::cout << "Resulting sift points are of size: " << cloud_temp->size () <<std::endl;
  pcl::io::savePCDFileASCII("sift_points.pcd", *cloud_temp);

  

  // Visualization of keypoints along with the original cloud
  pcl::visualization::PCLVisualizer viewer("PCL Viewer");
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler (cloud_temp, 0, 255, 0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> cloud_color_handler (cloud, 255, 255, 0);
  viewer.setBackgroundColor( 0.0, 0.0, 0.0 );
  viewer.addPointCloud(cloud, "cloud");
  viewer.addPointCloud(cloud_temp, keypoints_color_handler, "keypoints");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints");
  
  while(!viewer.wasStopped ())
  {
  viewer.spinOnce ();
  }


  
  return 0;
  
}

/*
int testcgalRegistartion(int argc, char** argv){
  std::string pathToBackground = argv[1];
  std::string pathToPlant = argv[2];

  //pcl::PointCloud<PointTypePCL>::Ptr cloudPlant(new pcl::PointCloud<PointTypePCL>);
  //pcl::PointCloud<PointTypePCL>::Ptr cloudBackground(new pcl::PointCloud<PointTypePCL>);

  pcl::PointCloud<PointTypePCL>::Ptr cloudPlant(new pcl::PointCloud<PointTypePCL>);
  pcl::PointCloud<PointTypePCL>::Ptr cloudBackground(new pcl::PointCloud<PointTypePCL>);

  if( pcl::io::loadPLYFile(pathToPlant, *cloudPlant) == -1){

    PCL_ERROR ("Couldn't read ply file\n");
    return (-1);

  }
  if( pcl::io::loadPLYFile(pathToBackground, *cloudBackground) == -1){

    PCL_ERROR ("Couldn't read ply file\n");
    return (-1);

  }
  registerPointCloudsCGAL(cloudBackground, cloudPlant);

  debug_showCombinedCloud(cloudBackground, cloudPlant, "CGAL Matching");
}

void testKeyPoints(int argc, char** argv){
  std::string pathToBackground = argv[1];
  std::string pathToPlant = argv[2];

  pcl::PointCloud<PointTypeRegistration>::Ptr cloudPlant(new pcl::PointCloud<PointTypeRegistration>);
  pcl::PointCloud<PointTypeRegistration>::Ptr cloudBackground(new pcl::PointCloud<PointTypeRegistration>);

  if( pcl::io::loadPLYFile(pathToPlant, *cloudPlant) == -1){

    PCL_ERROR ("Couldn't read ply file\n");
    return;

  }
  if( pcl::io::loadPLYFile(pathToBackground, *cloudBackground) == -1){

    PCL_ERROR ("Couldn't read ply file\n");
    return;

  }

  registerClouds(cloudPlant, cloudBackground);
  //testSIFTKeypoints(cloudPlant);
  //keyPointTest(cloudPlant);
}*/