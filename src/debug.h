#pragma once
#include "definitions.h"

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/common/centroid.h>

#include <thread>
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>

using namespace std::chrono_literals;

bool update;
bool vis_started = false;
boost::mutex updateModelMutex;
pcl::visualization::PCLVisualizer::Ptr viewer;
boost::thread t_viz;

void visualize()  
{  
    std::cout << "Visualization started" << std::endl;
    
    boost::shared_ptr<pcl::visualization::PCLVisualizer> tmp (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer = tmp;
    vis_started = true;
    //std::cout << "unlocking mutex" << std::endl;
    updateModelMutex.unlock();
    // prepare visualizer named "viewer"
    while (!viewer->wasStopped ())
    {
        // Get lock on the boolean update and check if cloud was updated
        updateModelMutex.lock();
          viewer->spinOnce (100);
        updateModelMutex.unlock();
        sleep(1);
    }   
    vis_started = false;
    viewer->close();
} 

boost::thread startVisualization(){
  
    return boost::thread(visualize);
  
}

void showCloud2(pcl::PointCloud<PointTypePCL>::Ptr cloud, std::string windowName, pcl::ModelCoefficients::Ptr coefficientsPlane = nullptr, bool showNormals = false){

  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer (windowName));
  viewer->setBackgroundColor (0, 0, 0);

  pcl::visualization::PointCloudColorHandlerRGBField<PointTypePCL> rgb(cloud);
  viewer->addPointCloud<PointTypePCL> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");

  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();

  //---------------------------------------
  //-----Add shapes at other locations-----
  //---------------------------------------
  
  if(coefficientsPlane != nullptr){

    pcl::ModelCoefficients coeffs;
    coeffs.values.push_back (0.0);
    coeffs.values.push_back (0.0);
    coeffs.values.push_back (1.0);
    coeffs.values.push_back (0.0);

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);
    
    cout << "centroid: " << centroid[0] << " " <<  centroid[1] << " " <<   centroid[2] << " " <<   centroid[3] << " \n";

    viewer->addPlane (*coefficientsPlane, centroid[0],centroid[1],centroid[2], "plane");
    viewer->addPlane (coeffs, centroid[0],centroid[1],centroid[2], "planeBase");
  }

  if(showNormals){
    viewer->addPointCloudNormals<PointTypePCL, PointTypePCL> (cloud, cloud, 10, 0.15, "normals"); 

  }
  
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    std::this_thread::sleep_for(100ms);
  }

}

void showCloud2(pcl::PointCloud<PointTypeRegistration>::Ptr cloud, std::string windowName, pcl::PointCloud<pcl::PointNormal>::Ptr normalCloud = nullptr){

  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer (windowName));
  viewer->setBackgroundColor (0, 0, 0);

  pcl::visualization::PointCloudColorHandlerRGBField<PointTypeRegistration> rgb(cloud);
  viewer->addPointCloud<PointTypeRegistration> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");

  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  
  if(normalCloud != nullptr){
    viewer->addPointCloudNormals<PointTypeRegistration, pcl::PointNormal> (cloud, normalCloud, 10, 0.3, "normals"); 
  }

  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    std::this_thread::sleep_for(100ms);
  }
  viewer->close();

}

void showCloud2(pcl::PointCloud<pcl::PointNormal>::Ptr cloud, std::string windowName, bool showNormals){

  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer (windowName));
  viewer->setBackgroundColor (0, 0, 0);

  viewer->addPointCloud<pcl::PointNormal> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");

  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  
  if(showNormals){
    viewer->addPointCloudNormals<pcl::PointNormal, pcl::PointNormal> (cloud, cloud, 10, 0.15, "normals"); 
  }

  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    std::this_thread::sleep_for(100ms);
  }
  viewer->close();

}


void debug_showCombinedCloud(pcl::PointCloud<PointTypePCL>::Ptr cloudA, pcl::PointCloud<PointTypePCL>::Ptr cloudB, std::string windowName){

  pcl::PointCloud<PointTypePCL>::Ptr combinedCloud (new pcl::PointCloud<PointTypePCL> ());

  *combinedCloud += *cloudA;
  *combinedCloud += *cloudB;
  
  showCloud2(combinedCloud, windowName);
}

void debug_showCombinedCloud(pcl::PointCloud<PointTypeRegistration>::Ptr cloudA, pcl::PointCloud<PointTypeRegistration>::Ptr cloudB, std::string windowName){

  pcl::PointCloud<PointTypeRegistration>::Ptr combinedCloud (new pcl::PointCloud<PointTypeRegistration> ());

  *combinedCloud += *cloudA;
  *combinedCloud += *cloudB;
  showCloud2(combinedCloud, windowName);
}

