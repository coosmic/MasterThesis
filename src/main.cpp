#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/filters/extract_indices.h>

#include <pcl/filters/statistical_outlier_removal.h>

#include <pcl/filters/radius_outlier_removal.h>

#include <pcl/common/common.h>

#include <pcl/filters/filter.h>

#define DEBUG true
//#define MIN_POINTS_IN_PLANE 20000

void showCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::string windowName){

  if(!DEBUG)
    return;

  pcl::visualization::CloudViewer viewer (windowName);
  viewer.showCloud (cloud);
  while (!viewer.wasStopped ())
  {
  }
}

void printMinMax(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud){

  pcl::PointXYZRGB minPt, maxPt;
  pcl::getMinMax3D (*cloud, minPt, maxPt);
  std::cout << "Max x: " << maxPt.x << std::endl;
  std::cout << "Max y: " << maxPt.y << std::endl;
  std::cout << "Max z: " << maxPt.z << std::endl;
  std::cout << "Min x: " << minPt.x << std::endl;
  std::cout << "Min y: " << minPt.y << std::endl;
  std::cout << "Min z: " << minPt.z << std::endl;

}

int main (int argc, char** argv)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

  if( pcl::io::loadPLYFile(argv[1], *cloud) == -1){

    PCL_ERROR ("Couldn't read ply file\n");
    return (-1);

  }

  printMinMax(cloud);
  //showCloud(cloud, "PlainCloud");


  ///////////////////
  // Remove Planes //
  ///////////////////

  long MIN_POINTS_IN_PLANE = cloud->size()*0.4;
  cout << "min points in plane: "<<MIN_POINTS_IN_PLANE<<endl;

  long pointsInPlane = 0;

  do{

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

    // Create the segmentation object
    pcl::SACSegmentation<pcl::PointXYZRGB> seg;
    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setDistanceThreshold (0.5);

    seg.setInputCloud (cloud);
    seg.segment (*inliers, *coefficients);

    pointsInPlane = inliers->indices.size();

    if(pointsInPlane > MIN_POINTS_IN_PLANE){
      pcl::ExtractIndices<pcl::PointXYZRGB> extract;
      extract.setInputCloud(cloud);
      extract.setIndices(inliers);
      extract.setNegative(true);
      extract.filter(*cloud);
    }
  
  }
  while(pointsInPlane > MIN_POINTS_IN_PLANE);

  showCloud(cloud, "PlaneFilter");

  //////////////////////////
  // Remove Noise Level 1 //
  //////////////////////////

  pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
    // build the filter
    outrem.setInputCloud(cloud);
    outrem.setRadiusSearch(0.8);
    outrem.setMinNeighborsInRadius (30);
    outrem.setKeepOrganized(true);
    // apply filter
    outrem.filter (*cloud);

  pcl::PointIndices::Ptr nanIndices (new pcl::PointIndices);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

  //showCloud(cloud, "Noise1");

  //////////////////////////
  // Remove Noise Level 2 //
  //////////////////////////

  pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem2;
    // build the filter
    outrem2.setInputCloud(cloud);
    outrem2.setRadiusSearch(3.0);
    outrem2.setMinNeighborsInRadius (500);
    outrem2.setKeepOrganized(true);
    // apply filter
    outrem2.filter (*cloud);

  //showCloud(cloud, "Noise2");

  return (0);
}
