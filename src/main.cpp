
#include "definitions.h"
#include "configuration.h"
#include "debug.h"
#include "registration_cgal.h"
#include "test.h"
#include "registration_pcl.h"
#include "utilities.h"
#include "utilities_io.h"

#include <iostream>
#include <thread>
#include <unordered_set>

#include <stdio.h>
#include <stdlib.h>

#include <chrono>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/ascii_io.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/common/io.h>
#include <pcl/common/pca.h>

#include <pcl/filters/filter.h>

#include <pcl/features/principal_curvatures.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/transformation_estimation_svd_scale.h>

#include "vtkPlaneSource.h"

#include <omp.h>

#include <Eigen/Dense>

#include <boost/program_options.hpp>


#define DEBUG true

//#define PointTypePCL pcl::PointXYZRGBNormal

//using namespace std::chrono_literals;

namespace po = boost::program_options;

void stemSegmentation3(pcl::PointCloud<PointTypePCL>::Ptr cloud){
  
  pcl::SACSegmentationFromNormals<PointTypePCL, PointTypePCL> seg;
  pcl::ModelCoefficients::Ptr coefficients_cylinder (new pcl::ModelCoefficients);
  
  pcl::ExtractIndices<PointTypePCL> extract, extract2;

  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_CYLINDER);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setNormalDistanceWeight (0.1);
  seg.setMaxIterations (2000);
  seg.setDistanceThreshold (0.075);
  seg.setRadiusLimits (0, 0.15);
  seg.setInputCloud (cloud);
  seg.setInputNormals (cloud);

  pcl::PointCloud<PointTypePCL>::Ptr cloud_labeld (new pcl::PointCloud<PointTypePCL> ());
  copyPointCloud(*cloud, *cloud_labeld);

  pcl::PointCloud<PointTypePCL>::Ptr cloud_cylinder (new pcl::PointCloud<PointTypePCL> ());

  while(true){

    pcl::PointIndices::Ptr inliers_cylinder (new pcl::PointIndices);
    // Obtain the cylinder inliers and coefficients
    seg.segment (*inliers_cylinder, *coefficients_cylinder);

    cout << "extracted cylinder with "<< inliers_cylinder->indices.size() <<" points"<<endl;

    if(inliers_cylinder->indices.size() == 0)
      break;

    extract.setInputCloud (cloud);
    extract.setIndices (inliers_cylinder);
    extract.setNegative (false);
    extract.filter (*cloud_cylinder);

    extract2.setInputCloud (cloud);
    extract2.setIndices (inliers_cylinder);
    extract2.setNegative(true);
    extract2.filter (*cloud);

  }

  for(int i=0; i< cloud_cylinder->points.size(); ++i){
    cloud_cylinder->points[i].r = 255;
    cloud_cylinder->points[i].g = 0;
    cloud_cylinder->points[i].b = 0;
  }

  //showCloud2(cloud_cylinder, "detected stems");
  *cloud_labeld += *cloud;
  *cloud_labeld += *cloud_cylinder;


  showCloud2(cloud_labeld, "other");
  //std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;

}

void stemSegmentation2(pcl::PointCloud<PointTypePCL>::Ptr cloud, float searchRadius, bool printInfos=true){
  pcl::PrincipalCurvaturesEstimation<PointTypePCL, PointTypePCL, pcl::PrincipalCurvatures> principalCurvaturesEstimation;
  principalCurvaturesEstimation.setInputCloud (cloud);
  principalCurvaturesEstimation.setInputNormals (cloud);

  pcl::search::KdTree<PointTypePCL>::Ptr tree (new pcl::search::KdTree<PointTypePCL>);
  principalCurvaturesEstimation.setSearchMethod (tree);
  principalCurvaturesEstimation.setRadiusSearch(searchRadius);

  // Actually compute the principal curvatures
  pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principalCurvatures (new pcl::PointCloud<pcl::PrincipalCurvatures> ());
  principalCurvaturesEstimation.compute (*principalCurvatures);

  pcl::PrincipalCurvatures descriptor = principalCurvatures->points[0];
  std::cout << descriptor << std::endl;

  if(printInfos){
    float pc1ScaleFactorToColor = 3.947358033 * 255;
    float minpc1 = FLT_MAX, maxpc1 = FLT_MIN, minpc2 = FLT_MAX, maxpc2 = FLT_MIN, maxpcx = FLT_MIN, minpcx = FLT_MAX, maxpcy = FLT_MIN, minpcy = FLT_MAX, maxpcz = FLT_MIN, minpcz = FLT_MAX;
    #pragma omp parallel for
    for(int i=0; i<principalCurvatures->points.size(); ++i){
      if(principalCurvatures->points[i].pc1 < minpc1)
        minpc1 = principalCurvatures->points[i].pc1;
      else if(principalCurvatures->points[i].pc1 > maxpc1)
        maxpc1 = principalCurvatures->points[i].pc1;

      if(principalCurvatures->points[i].pc2 < minpc2)
        minpc2 = principalCurvatures->points[i].pc2;
      else if(principalCurvatures->points[i].pc2 > maxpc2)
        maxpc2 = principalCurvatures->points[i].pc2;

      if(principalCurvatures->points[i].principal_curvature_x < minpcx)
        minpcx = principalCurvatures->points[i].principal_curvature_x;
      else if(principalCurvatures->points[i].principal_curvature_x > maxpcx)
        maxpcx = principalCurvatures->points[i].principal_curvature_x;

      if(principalCurvatures->points[i].principal_curvature_y < minpcy)
        minpcy = principalCurvatures->points[i].principal_curvature_y;
      else if(principalCurvatures->points[i].principal_curvature_y > maxpcy)
        maxpcy = principalCurvatures->points[i].principal_curvature_y;

      if(principalCurvatures->points[i].principal_curvature_z < minpcz)
        minpcz = principalCurvatures->points[i].principal_curvature_z;
      else if(principalCurvatures->points[i].principal_curvature_z > maxpcz)
        maxpcz = principalCurvatures->points[i].principal_curvature_z;

      //cloud->points[i].r = 0;
      //cloud->points[i].g = 0;
      //cloud->points[i].b = max(255 * (principalCurvatures->points[i].principal_curvature_x), max(255 * principalCurvatures->points[i].principal_curvature_y, 255 * principalCurvatures->points[i].principal_curvature_z));
      //std::cout << i << " pc1: "<< principalCurvatures->points[i].pc1 << " pc2: " << principalCurvatures->points[i].pc2 << std::endl;
      /*if(principalCurvatures->points[i].pc1 * principalCurvatures->points[i].pc2 > 0.0){
        cloud->points[i].r = 0;
        cloud->points[i].g = 0;
        cloud->points[i].b = 255;
      }

      if(principalCurvatures->points[i].pc1 * principalCurvatures->points[i].pc2 < 0.0){
        cloud->points[i].r = 255;
        cloud->points[i].g = 0;
        cloud->points[i].b = 0;
      }

      if(principalCurvatures->points[i].pc1 <= 0.001 && principalCurvatures->points[i].pc2 <= 0.0001 && principalCurvatures->points[i].pc1 >= 0.001 && principalCurvatures->points[i].pc2 >= 0.0001){
        cloud->points[i].r = 0;
        cloud->points[i].g = 255;
        cloud->points[i].b = 0;
      }

      if(principalCurvatures->points[i].pc1 - principalCurvatures->points[i].pc2 <= 0.01 ){
        cloud->points[i].r = 255;
        cloud->points[i].g = 255;
        cloud->points[i].b = 0;
      }*/
      cloud->points[i].r = (uint8_t)((principalCurvatures->points[i].pc1 - 0.192002f) * pc1ScaleFactorToColor);
      cloud->points[i].g = 0;
      cloud->points[i].b = 0;
    }

    cout<< "maxpc1: "<<maxpc1 << "\nminpc1: "<<minpc1<<"\nmaxpc2: "<<maxpc2<<"\nminpc2: "<<minpc2<<endl;

    cout<< "maxpcx: "<<maxpcx << "\nminpcx: "<<minpcx<<"\nmaxpcy: "<<maxpcy<<"\nminpcy: "<<minpcy<<"\nmaxpcz: "<<maxpcz<<"\nminpcz: "<<minpcz<<endl;
    showCloud2(cloud, "intensity");
  }


  /*std::string input = "asdf";
  cout << "enter float: "<<endl;
  cin >> input;
  while(input != "q"){
    float threshold = stof(input);

    for(int i=0; i<principalCurvatures->points.size(); ++i){
      //float meanCurvature = (principalCurvatures->points[i].pc1 + principalCurvatures->points[i].pc2) * 0.5;

      if(principalCurvatures->points[i].pc1  > threshold){
        cloud->points[i].r = 255;
        cloud->points[i].g = 0;
        cloud->points[i].b = 0;
      } else {
        cloud->points[i].r = 0;
        cloud->points[i].g = 0;
        cloud->points[i].b = 255;
      }
    }

    showCloud2(cloud, "segmentation", nullptr, true);

    cout << "enter float: "<<endl;
    cin >> input;

  }*/

  std::string input = "asdf";
  cout << "enter float: "<<endl;
  cin >> input;
  while(input != "q"){
    float threshold = stof(input);

    //#pragma omp parallel for
    for(int i=0; i<principalCurvatures->points.size(); ++i){
      float linearity_i = principalCurvatures->points[i].pc1;
      //float linearity_i = (principalCurvatures->points[i].pc1 - principalCurvatures->points[i].pc2) / principalCurvatures->points[i].pc1;
      //float linearity_i = max(principalCurvatures->points[i].principal_curvature_x, max(principalCurvatures->points[i].principal_curvature_y, principalCurvatures->points[i].principal_curvature_z));
      if(linearity_i > threshold){
        cloud->points[i].r = 255;
        cloud->points[i].g = 0;
        cloud->points[i].b = 0;
      } else {
        cloud->points[i].r = 0;
        cloud->points[i].g = 0;
        cloud->points[i].b = 255;
      }
    }

    showCloud2(cloud, "segmentation");

    cout << "enter float: "<<endl;
    cin >> input;

  }
  
}

void stemSegementation(pcl::PointCloud<PointTypePCL>::Ptr cloud, float searchRadius){

  pcl::KdTreeFLANN<PointTypePCL> kdtree;
  kdtree.setInputCloud (cloud);

  int pointsInCloud = cloud->points.size();

  //#pragma omp parallel num_threads(24)
  //#pragma omp target teams distribute parallel for collapse(2)
  #pragma omp parallel for 
  for(int i=0; i< pointsInCloud; i++){
    std::vector<int> neighborIndices; //to store index of surrounding points 
    std::vector<float> pointRadiusSquaredDistance; // to store distance to surrounding

    PointTypePCL searchPoint = cloud->points[i];
    kdtree.radiusSearch(searchPoint, searchRadius, neighborIndices, pointRadiusSquaredDistance);

    int numberOfNeighbors = neighborIndices.size();

    //cout << "found " << pointIdxRadiusSearch.size() << " neighbor point(s) for point " << searchPoint <<endl;

    Eigen::Matrix3f identity = Eigen::Matrix3f::Identity();

    Eigen::Vector3f normal_i = searchPoint.getNormalVector3fMap();
    Eigen::Matrix3f kernel = (identity - (normal_i * normal_i.transpose()) );

    std::vector<Eigen::Vector3f> list_m_j(numberOfNeighbors);
    Eigen::Vector3f m_mean = Eigen::Vector3f::Zero();
    for(int j=0; j<numberOfNeighbors; j++){
      Eigen::Vector3f m_j = kernel * cloud->points[neighborIndices[j]].getNormalVector3fMap();
      list_m_j[j] = m_j;
      m_mean += m_j;
    }
    m_mean /= numberOfNeighbors;
    Eigen::Matrix3f convarianz_i = Eigen::Matrix3f::Zero();
    for(int j=0; j<numberOfNeighbors; j++){
      Eigen::Vector3f tmp = list_m_j[j] - m_mean;
      convarianz_i += tmp * tmp.transpose();
    }
    convarianz_i /= numberOfNeighbors;
    Eigen::Vector3cf eigenvalues_i = convarianz_i.eigenvalues();

    cout << i << "/" << pointsInCloud << ": min curv "<<eigenvalues_i.y()<< " max curv "<< eigenvalues_i.z()<< endl;
    
  }
}



int text_id =0;
void mouseEventOccurred (const pcl::visualization::MouseEvent &event,
                         void* viewer_void)
{
  pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
  if (event.getButton () == pcl::visualization::MouseEvent::LeftButton &&
      event.getType () == pcl::visualization::MouseEvent::MouseButtonRelease)
  {
    std::cout << "Left mouse button released at position (" << event.getX () << ", " << event.getY () << ")" << std::endl;

    char str[512];
    sprintf (str, "text#%03d", text_id ++);
    viewer->addText ("clicked here", event.getX (), event.getY (), str);
  }
}

void printMinMax(pcl::PointCloud<PointTypePCL>::Ptr cloud){

  PointTypePCL minPt, maxPt;
  pcl::getMinMax3D (*cloud, minPt, maxPt);
  std::cout << "Max x: " << maxPt.x << std::endl;
  std::cout << "Max y: " << maxPt.y << std::endl;
  std::cout << "Max z: " << maxPt.z << std::endl;
  std::cout << "Min x: " << minPt.x << std::endl;
  std::cout << "Min y: " << minPt.y << std::endl;
  std::cout << "Min z: " << minPt.z << std::endl;

}

int main1(int argc, char** argv){
  pcl::PointCloud<PointTypePCL>::Ptr cloud(new pcl::PointCloud<PointTypePCL>);

  std::string pathToFile = argv[1];

  if(pathToFile.substr(pathToFile.find_last_of(".") + 1) == "ply"){
    if( pcl::io::loadPLYFile(pathToFile, *cloud) == -1){

      PCL_ERROR ("Couldn't read ply file\n");
      return (-1);

    }
  } else if(pathToFile.substr(pathToFile.find_last_of(".") + 1) == "txt"){
    if(!loadAsciCloud(pathToFile, cloud)){
      PCL_ERROR ("Couldn't read txt file\n");
      return (-1);
    }
    showCloud2(cloud, "raw acsi cloud");
      
  } else
    return -1;

  ///////////////////
  // Remove Planes //
  ///////////////////

  pcl::ModelCoefficients::Ptr coefficients = planeFilter(cloud);

  //showCloud2(cloud, "PlaneFilter", coefficients);

  //////////////////////////
  // Remove Noise Level 1 //
  //////////////////////////
  noiseFilter(cloud, 30, 0.8);

  //showCloud(cloud, "Noise1");

  //////////////////////////
  // Remove Noise Level 2 //
  //////////////////////////

  noiseFilter(cloud, 500, 3.0);

  //////////////////////////
  // Remove Noise Level 3 //
  //////////////////////////

  noiseFilter(cloud, 1000, 10.0);

  rotateCloud(cloud, coefficients);
  //showCloud2(cloud, "Noise2", nullptr, true);

  printMinMax(cloud);
  //stemSegementation(cloud, 0.25);
  stemSegmentation2(cloud, 0.25);
  //stemSegmentation3(cloud);

  return (0);
}



void matchClouds(pcl::PointCloud<PointTypePCL>::Ptr cloudA, pcl::PointCloud<PointTypePCL>::Ptr cloudB){
  pcl::ModelCoefficients::Ptr coefficientsA (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliersA (new pcl::PointIndices);
  findPlaneInCloud(cloudA, coefficientsA, inliersA);

  //pcl::ModelCoefficients::Ptr coefficientsB (new pcl::ModelCoefficients);
  //pcl::PointIndices::Ptr inliersB(new pcl::PointIndices);
  //findPlaneInCloud(cloudB, coefficientsB, inliersB);

  rotateCloud(cloudA, coefficientsA);
  //rotateCloud(cloudB, coefficientsB);

  //debug_showCombinedCloud(cloudA, cloudB, "rotated Clouds");
  cout << "Transform clouds to same size..." <<endl;
  transformCloudsToSameSize(cloudA, cloudB);
  //debug_showCombinedCloud(cloudA, cloudB, "prescaled Clouds");

  

  cout << "extracting center of clouds..." <<endl;
  extractCenterOfCloud(cloudA, 0.3);
  extractCenterOfCloud(cloudB, 0.3);
  //debug_showCombinedCloud(cloudA, cloudB, "centered Clouds");

  cout << "Transform clouds to same size..." <<endl;
  transformCloudsToSameSize(cloudA, cloudB);
  centerCloud(cloudA);
  debug_showCombinedCloud(cloudA, cloudB, "rescaled src Cloud");

  cout << "downsampling clouds..." <<endl;
  downsampleCloud(cloudA, 0.0075f);
  downsampleCloud(cloudB, 0.0075f);
  cout << "cloudA Points: "<<cloudA->size()<<endl; 
  //debug_showCombinedCloud(cloudA, cloudB, "downsampled Clouds");

  cout << "noise filtering clouds..." <<endl;
  noiseFilter(cloudA);
  noiseFilter(cloudB);
  debug_showCombinedCloud(cloudA, cloudB, "noise filtered Clouds");

  registrationPipeline(cloudA, cloudB, true, true, true, false, true); 

  cout << "Transform clouds to same size..." <<endl;
  transformCloudsToSameSize(cloudA, cloudB);
  centerCloud(cloudA);
  debug_showCombinedCloud(cloudA, cloudB, "rescaled src Cloud");

  

}

void backgroundPreparation(Cloud::Ptr backgroundCloud){
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  findPlaneInCloud(backgroundCloud, coefficients, inliers);

  rotateCloud(backgroundCloud, coefficients);

  transformCloudToUnitSize(backgroundCloud, 5.0);
  centerCloud(backgroundCloud);
  //showCloud2(backgroundCloud, "Background");
}

int backgroundRemovalPipeline(int argc, char** argv){
  boost::thread t = startVisualization();
  /*std::string pathToFolder = argv[1];

  std::string pathToBackground = pathToFolder+"/background2.ply";
  std::string pathToPlant = pathToFolder+"/t1.ply";*/

  std::string pathToBackground = argv[1];
  std::string pathToPlant = argv[2];

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

  backgroundPreparation(cloudBackground);

  setColorChannelExclusive(cloudPlant, ColorChannel::b, 255);
  setColorChannelExclusive(cloudBackground, ColorChannel::r, 255);

  //debug_showCombinedCloud(cloudPlant, cloudBackground, "ChannelSwap");

  matchClouds(cloudPlant, cloudBackground);
  debug_showCombinedCloud(cloudBackground, cloudPlant, "Matched Clouds");
  substractCloudFromOtherCloud(cloudBackground, cloudPlant, 0.05);

  //showCloud2(cloudPlant, "Cloud without Background");

  stemSegmentation2(cloudPlant, 3.0);

  showCloud2(cloudPlant, "Classified Cloud");
  t.join();
  return 0;

}

int convertToShapenetFormat2(po::variables_map vm){

  if(!vm.count("PointCloudName")){
    cout << "Missing Parameter PointCloudName\n";
    return 1;
  }

  if(!vm.count("in")){
    cout << "Missing parameter in\n";
    return 1;
  }
  if(!vm.count("out")){
    cout << "Missing parameter out\n";
    return 1;
  }

  std::string namePly = vm["PointCloudName"].as<std::string>()+".ply";
  std::string nameTxt = vm["PointCloudName"].as<std::string>()+".txt";
  std::string pathToShapenetFormatResult = vm["out"].as<std::string>();
  std::string pathToPlant = vm["in"].as<std::string>()+namePly;

  pcl::PointCloud<PointTypePCL>::Ptr cloudPlant(new pcl::PointCloud<PointTypePCL>);

  if( pcl::io::loadPLYFile(pathToPlant, *cloudPlant) == -1){

    PCL_ERROR ("Couldn't read ply file\n");
    return (-1);

  }

  pcl::ModelCoefficients::Ptr coefficientsA (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliersA (new pcl::PointIndices);
  findPlaneInCloud(cloudPlant, coefficientsA, inliersA);
  rotateCloud(cloudPlant, coefficientsA);
  //showCloud2(cloudPlant, "rotation");

  //showCloud2(cloudPlant, "Labeled Cloud");
  removeBackgroundPointsShapenet(cloudPlant);
  //showCloud2(cloudPlant, "Cloud Without Background");
  Eigen::Matrix4f t,s;
  transformToShapenetFormat(cloudPlant, t,s);
  //showCloud2(cloudPlant, "Shapenet Formatted Cloud");

  writeShapenetFormat2(cloudPlant, pathToShapenetFormatResult, nameTxt);

  return 0;
}

int convertToShapenetFormat(po::variables_map vm){

  if(!vm.count("in")){
    cout << "Missing parameter in\n";
    return 1;
  }
  if(!vm.count("out")){
    cout << "Missing parameter out\n";
    return 1;
  }
  if(!vm.count("RemoveBackground")){
    cout << "Missing parameter RemoveBackground\n";
    return 1;
  }
  if(!vm.count("CenterOnly")){
    cout << "Missing parameter CenterOnly\n";
    return 1;
  }

  int maxSubsample = 20;
  if(vm.count("MaxSubsample")){
    maxSubsample = vm["MaxSubsample"].as<int>();
  }

  int numOfPointsPerSubsample = 16384;
  if(vm.count("SubsamplePointCount")){
    numOfPointsPerSubsample = vm["SubsamplePointCount"].as<int>();
  }

  std::string pathToPlant = vm["in"].as<std::string>();
  std::string pathToShapenetFormatResult = vm["out"].as<std::string>();
  bool removeBackground = vm["RemoveBackground"].as<bool>();
  bool centerOnly = vm["CenterOnly"].as<bool>();

  cout << "Converting file saved under "<< pathToPlant  << " to Shapenet format\n";

  pcl::PointCloud<PointTypePCL>::Ptr cloudPlant(new pcl::PointCloud<PointTypePCL>);

  if( pcl::io::loadPLYFile(pathToPlant, *cloudPlant) == -1){
    PCL_ERROR ("Couldn't read ply file\n");
    return (-1);
  }
  
  if(!vm["NoPlaneAlignment"].as<bool>()){
    std::cout << "Plane Alignment started" <<std::endl;
    pcl::ModelCoefficients::Ptr coefficientsA (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliersA (new pcl::PointIndices);
    findPlaneInCloud(cloudPlant, coefficientsA, inliersA);
    rotateCloud(cloudPlant, coefficientsA);
  }

  //showCloud2(cloudPlant, "rotation");

  //showCloud2(cloudPlant, "Labeled Cloud");
  if(removeBackground){
    removeBackgroundPointsShapenet(cloudPlant);
  } else if(centerOnly) {
    bool containsBG = false;
    for(int i=0; i<cloudPlant->size(); i++){
      int colorCode = colorToCode(cloudPlant->points[i]);
      if(colorCode == BackgroundLabel){
        containsBG = true;
        break;
      }
        
    }
    if(containsBG)
      extractCenterOfCloud(cloudPlant, 0.3);
  }
  
  //showCloud2(cloudPlant, "Cloud Without Background");
  Eigen::Matrix4f t,s;
  transformToShapenetFormat(cloudPlant, t,s);
  //showCloud2(cloudPlant, "Shapenet Formatted Cloud");

  int createdSubsamplesCount = 0;
  while(cloudPlant->size() > numOfPointsPerSubsample && createdSubsamplesCount < maxSubsample){
    Cloud::Ptr subsampledCloud = subSampleCloudRandom(cloudPlant, numOfPointsPerSubsample);
    createdSubsamplesCount++;

    assert(subsampledCloud->size() == numOfPointsPerSubsample);
    if(vm["RotateRandom"].as<bool>()){
      Eigen::Affine3f rotation = Eigen::Affine3f::Identity();
      float theta = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      rotation.rotate(Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitZ()));
      theta = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      rotation.rotate(Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitX()));
      theta = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      rotation.rotate(Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitY()));
      pcl::transformPointCloud (*subsampledCloud, *subsampledCloud, rotation);
    }
    writeShapenetFormat(subsampledCloud, pathToShapenetFormatResult+"SS"+std::to_string(createdSubsamplesCount), removeBackground);

    //cout << "remaining points in org cloud: "<<std::to_string(cloudPlant->size())<<endl;
  }


  //writeShapenetFormat(cloudPlant, pathToShapenetFormatResult, false);
  pcl::io::savePLYFileBinary(pathToShapenetFormatResult+".ply", *cloudPlant);

  return 0;
}

int iterativeScaleRegistration(po::variables_map vm){

  if(!vm.count("SourceCloudPath")){
    cout << "Missing Parameter SourceCloudPath\n";
    return 1;
  }

  if(!vm.count("TargetCloudPath")){
    cout << "Missing Parameter TargetCloudPath\n";
    return 1;
  }

  if(!vm.count("SubsamplePointCount")){
    cout << "Missing Parameter SubsamplePointCount\n";
    return 1;
  }

  std::string srcCloudPath = vm["SourceCloudPath"].as<std::string>();
  std::string tgtCloudPath = vm["TargetCloudPath"].as<std::string>();
  int subsampleCount = vm["SubsamplePointCount"].as<int>();

  pcl::PointCloud<PointTypePCL>::Ptr cloudSrc(new pcl::PointCloud<PointTypePCL>);

  if( pcl::io::loadPCDFile(srcCloudPath, *cloudSrc) == -1){

    PCL_ERROR ("Couldn't read pcd file\n");
    return (-1);
  }

  pcl::PointCloud<PointTypePCL>::Ptr cloudTgt(new pcl::PointCloud<PointTypePCL>);

  if( pcl::io::loadPLYFile(tgtCloudPath, *cloudTgt) == -1){

    PCL_ERROR ("Couldn't read ply file\n");
    return (-1);
  }

  pcl::ModelCoefficients::Ptr coefficientsA (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliersA (new pcl::PointIndices);
  findPlaneInCloud(cloudSrc, coefficientsA, inliersA);
  rotateCloud(cloudSrc, coefficientsA);
  //removePointsInCloud(cloudSrc, inliersA);
  findPlaneInCloud(cloudTgt, coefficientsA, inliersA);
  rotateCloud(cloudTgt, coefficientsA);
  //removePointsInCloud(cloudTgt, inliersA);

  

  extractCenterOfCloud(cloudSrc, 0.3);
  extractCenterOfCloud(cloudTgt, 0.3);

  Cloud::Ptr backgroundSrc = removeBackgroundPointsShapenet(cloudSrc);
  //showCloud2(backgroundSrc, "Background Src");

  pcl::PointCloud<PointTypePCL>::Ptr cloudSrcSubsampled = subSampleCloudRandom(backgroundSrc, subsampleCount);
  pcl::PointCloud<PointTypePCL>::Ptr cloudTgtSubsampled = subSampleCloudRandom(cloudTgt, subsampleCount);
  setColorChannelExclusive(cloudSrcSubsampled, ColorChannel::r, 255);
  setColorChannelExclusive(cloudSrcSubsampled, ColorChannel::g, 0);
  setColorChannelExclusive(cloudSrcSubsampled, ColorChannel::b, 0);

  setColorChannelExclusive(cloudTgtSubsampled, ColorChannel::r, 0);
  setColorChannelExclusive(cloudTgtSubsampled, ColorChannel::g, 0);
  setColorChannelExclusive(cloudTgtSubsampled, ColorChannel::b, 255);

  Eigen::Matrix4f ts,ss,tt,st;
  transformToShapenetFormat(cloudTgtSubsampled, tt,st);
  transformToShapenetFormat(cloudSrcSubsampled, ts,ss); 
  //debug_showCombinedCloud(cloudSrcSubsampled, cloudTgtSubsampled, "ShapenetFormat");

  double bestScore = DBL_MAX;
  float scaleFactor = 0.01;
  int bestScaleStep = 0;
  Eigen::Matrix4f bestTransformation;
  Eigen::Matrix4f scaleMatrix = Eigen::Matrix4f::Zero();
  scaleMatrix(0,0) = scaleFactor;
  scaleMatrix(1,1) = scaleFactor;
  scaleMatrix(2,2) = scaleFactor;
  scaleMatrix(3,3) = 1.0;
  for(int i=50; i<200; ++i){
    pcl::PointCloud<PointTypePCL>::Ptr cloudSrcSubsampledICP = pcl::PointCloud<PointTypePCL>().makeShared();
    pcl::IterativeClosestPoint<PointTypePCL, PointTypePCL> icp;
    pcl::transformPointCloud (*cloudSrcSubsampled, *cloudSrcSubsampledICP, scaleMatrix*(float)i);
    icp.setRANSACOutlierRejectionThreshold(0.001);
    icp.setInputSource(cloudTgtSubsampled);
    icp.setInputTarget(cloudSrcSubsampledICP);

    pcl::PointCloud<PointTypePCL> Unused;
    icp.align(Unused);

    pcl::transformPointCloud (*cloudSrcSubsampledICP, *cloudSrcSubsampledICP, icp.getFinalTransformation().inverse());
    //if(i % 10 == 0)
      //debug_showCombinedCloud(cloudSrcSubsampledICP, cloudTgtSubsampled, "ShapenetFormat");
    double score = icp.getFitnessScore(0.1);
    std::cout<< i << " Score: " << score << " " << icp.getFitnessScore(0.9) << std::endl;
    if(score < bestScore){
      bestTransformation = icp.getFinalTransformation().inverse();
      bestScore = score;
      bestScaleStep = i;
    }
  }

  pcl::PointCloud<PointTypePCL>::Ptr cloudSrcResultTmp = pcl::PointCloud<PointTypePCL>().makeShared();

  //transform subsamped src to subsampeled target
  pcl::transformPointCloud (*cloudSrcSubsampled, *cloudSrcSubsampled, scaleMatrix*(float)bestScaleStep);
  pcl::transformPointCloud (*cloudSrcSubsampled, *cloudSrcSubsampled, bestTransformation);
  debug_showCombinedCloud(cloudSrcSubsampled, cloudTgtSubsampled, "Subsample ICP Result with best Scale");

  //transform src to subsampled target
  pcl::transformPointCloud (*cloudSrc, *cloudSrcResultTmp, ts);
  pcl::transformPointCloud (*cloudSrcResultTmp, *cloudSrcResultTmp, ss);
  pcl::transformPointCloud (*cloudSrcResultTmp, *cloudSrcResultTmp, scaleMatrix*(float)bestScaleStep);
  pcl::transformPointCloud (*cloudSrcResultTmp, *cloudSrcResultTmp, bestTransformation);
  //transform target to subsampled target
  pcl::PointCloud<PointTypePCL>::Ptr cloudTargetResultTmp = pcl::PointCloud<PointTypePCL>().makeShared();
  pcl::transformPointCloud (*cloudTgt, *cloudTargetResultTmp, tt);
  pcl::transformPointCloud (*cloudTargetResultTmp, *cloudTargetResultTmp, st);
  debug_showCombinedCloud(cloudSrcResultTmp, cloudTargetResultTmp, "ICP Result with best Scale");

  /*pcl::IterativeClosestPoint<PointTypePCL, PointTypePCL> icp;
  icp.setInputSource(cloudSrcSubsampled);
  icp.setInputTarget(cloudTgtSubsampled);

  pcl::PointCloud<PointTypePCL> Unused;
  icp.align(Unused);
  pcl::transformPointCloud (*cloudSrcResultTmp, *cloudSrcResultTmp, icp.getFinalTransformation());
  debug_showCombinedCloud(cloudSrcResultTmp, cloudTargetResultTmp, "Final ICP Result");*/

  return 0;
}

int showCloudWithNormals(po::variables_map vm){
  if(!vm.count("PointCloudPlant")){
    cout << "Missing Parameter PointCloudPlant\n";
    return 1;
  }

  std::string pathToPlant = vm["PointCloudPlant"].as<std::string>();

  pcl::PointCloud<PointTypePCL>::Ptr cloudPlant(new pcl::PointCloud<PointTypePCL>);

  if( pcl::io::loadPLYFile(pathToPlant, *cloudPlant) == -1){

    PCL_ERROR ("Couldn't read ply file\n");
    return (-1);

  }

  centerCloud(cloudPlant);

  showCloud2(cloudPlant, "Cloud with normals", nullptr, true);

  return 0;

}

int convertToRegistrationFormat(po::variables_map vm){
  if(!vm.count("in")){
    cout << "Missing Parameter in\n";
    return 1;
  }

  if(!vm.count("out")){
    cout << "Missing Parameter out\n";
    return 1;
  }

  std::string inPath = vm["in"].as<std::string>();
  std::string outPath = vm["out"].as<std::string>();

  pcl::PointCloud<PointTypePCL>::Ptr cloudSrc(new pcl::PointCloud<PointTypePCL>);

  if( pcl::io::loadPLYFile(inPath, *cloudSrc) == -1){

    PCL_ERROR ("Couldn't read ply file\n");
    return (-1);

  }

  pcl::ModelCoefficients::Ptr coefficientsA (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliersA (new pcl::PointIndices);
  findPlaneInCloud(cloudSrc, coefficientsA, inliersA);
  rotateCloud(cloudSrc, coefficientsA);

  extractCenterOfCloud(cloudSrc, 0.3);

  int numOfPoints = 1024;
  if(vm.count("SubsamplePointCount")){
    numOfPoints = vm["SubsamplePointCount"].as<int>();
  }

  pcl::PointCloud<PointTypePCL>::Ptr cloudSrcSubsampled = subSampleCloudRandom(cloudSrc, numOfPoints);

  Eigen::Matrix4f t,s;
  transformToShapenetFormat(cloudSrcSubsampled, t,s);

  writeRegistrationFormat(cloudSrcSubsampled, outPath);

  return 0;
}

int convertBothToRegistrationFormat(po::variables_map vm){
  if(!vm.count("SourceCloudPath")){
    cout << "Missing Parameter SourceCloudPath\n";
    return 1;
  }

  if(!vm.count("OutputFolder")){
    cout << "Missing Parameter OutputFolder\n";
    return 1;
  }

  std::string srcCloudPath = vm["SourceCloudPath"].as<std::string>();
  std::string tgtCloudPath = vm["TargetCloudPath"].as<std::string>();
  std::string outFolderPath = vm["OutputFolder"].as<std::string>();

  pcl::PointCloud<PointTypePCL>::Ptr cloudSrc(new pcl::PointCloud<PointTypePCL>);

  if( pcl::io::loadPLYFile(srcCloudPath, *cloudSrc) == -1){

    PCL_ERROR ("Couldn't read ply file\n");
    return (-1);

  }

  pcl::PointCloud<PointTypePCL>::Ptr cloudTgt(new pcl::PointCloud<PointTypePCL>);

  if( pcl::io::loadPLYFile(tgtCloudPath, *cloudTgt) == -1){

    PCL_ERROR ("Couldn't read ply file\n");
    return (-1);

  }

  pcl::ModelCoefficients::Ptr coefficientsA (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliersA (new pcl::PointIndices);
  findPlaneInCloud(cloudSrc, coefficientsA, inliersA);
  rotateCloud(cloudSrc, coefficientsA);
  findPlaneInCloud(cloudTgt, coefficientsA, inliersA);
  rotateCloud(cloudTgt, coefficientsA);

  extractCenterOfCloud(cloudSrc, 0.3);
  extractCenterOfCloud(cloudTgt, 0.3);

  pcl::PointCloud<PointTypePCL>::Ptr cloudSrcSubsampled = subSampleCloudRandom(cloudSrc, 1024);
  pcl::PointCloud<PointTypePCL>::Ptr cloudTgtSubsampled = subSampleCloudRandom(cloudTgt, 1024);

  Eigen::Matrix4f t,s;
  transformToShapenetFormat(cloudSrcSubsampled, t,s);
  transformToShapenetFormat(cloudTgtSubsampled, t,s);
  debug_showCombinedCloud(cloudSrcSubsampled, cloudTgtSubsampled, "ShapenetFormat");

  //TODO: Fix and add to busybox
  //writeRegistrationFormat(cloudSrcSubsampled, outFolderPath, "SrcCloud");
  //writeRegistrationFormat(cloudTgtSubsampled, outFolderPath, "TgtCloud");

  return 0;
}

int computeAndShowNormals(po::variables_map vm){
  if(!vm.count("PointCloudPlant")){
    cout << "Missing Parameter PointCloudPlant\n";
    return 1;
  }

  if(!vm.count("NormalEsitmationSearchRadius")){
    cout << "Missing Parameter NormalEsitmationSearchRadius\n";
    return 1;
  }

  std::string pathToPlant = vm["PointCloudPlant"].as<std::string>();

  pcl::PointCloud<PointTypePCL>::Ptr cloudPlant(new pcl::PointCloud<PointTypePCL>);

  if( pcl::io::loadPLYFile(pathToPlant, *cloudPlant) == -1){

    PCL_ERROR ("Couldn't read ply file\n");
    return (-1);

  }

  float normalEstiamtionRadius = vm["NormalEsitmationSearchRadius"].as<float>();
  cout << "Normal estimation will be done with search radius "<<normalEstiamtionRadius <<endl;
  pcl::NormalEstimation<PointTypePCL, PointTypePCL> ne;
  pcl::search::KdTree<PointTypePCL>::Ptr tree_xyz (new pcl::search::KdTree<PointTypePCL>());
  ne.setInputCloud(cloudPlant);
  ne.setSearchMethod(tree_xyz);
  //ne.setRadiusSearch(normalEstiamtionRadius);
  ne.setKSearch(8);
  ne.compute(*cloudPlant);

  centerCloud(cloudPlant);
  showCloud2(cloudPlant, "Cloud with estimated normals", nullptr, true);

  return 0;

}

int backgroundRemovalPipeline(po::variables_map vm){
  if(!vm.count("SourceCloudPath")){
    cout << "Missing Parameter SourceCloudPath\n";
    return 1;
  }

  if(!vm.count("TargetCloudPath")){
    cout << "Missing Parameter TargetCloudPath\n";
    return 1;
  }

  if(!vm.count("OutputFolder")){
    cout << "Missing Parameter OutputFolder\n";
    return 1;
  }

  float searchRadius = 0.0125;
  if(vm.count("SearchRadius")){
    searchRadius = vm["SearchRadius"].as<float>();
  }

  std::string srcCloudPath = vm["SourceCloudPath"].as<std::string>();
  std::string tgtCloudPath = vm["TargetCloudPath"].as<std::string>();
  std::string outFolderPath = vm["OutputFolder"].as<std::string>();

  pcl::PointCloud<PointTypePCL>::Ptr cloudSrc(new pcl::PointCloud<PointTypePCL>);

  if( pcl::io::loadPCDFile(srcCloudPath, *cloudSrc) == -1){

    PCL_ERROR ("Couldn't read pcd file\n");
    return (-1);
  }

  pcl::PointCloud<PointTypePCL>::Ptr cloudTgt(new pcl::PointCloud<PointTypePCL>);

  if( pcl::io::loadPLYFile(tgtCloudPath, *cloudTgt) == -1){

    PCL_ERROR ("Couldn't read ply file\n");
    return (-1);
  }
  //showCloud2(cloudSrc, "Cloud with Background");
  std::cout << "removing background by shapnet label" << std::endl;
  Cloud::Ptr background = removeBackgroundPointsShapenet(cloudSrc);

  Cloud::Ptr cloudWithoutPlant = getPointsNearCloudFromOtherCloud(background, cloudTgt, searchRadius);
  pcl::io::savePLYFile(outFolderPath+"/CloudWithoutPlant.ply", *cloudWithoutPlant);

  //showCloud2(cloudSrc, "Cloud without Background");
  //showCloud2(background, "Background");
  //debug_showCombinedCloud(background, cloudTgt, "Background with cloudTgt");
  Cloud::Ptr cloudWithoutBackground = getPointsNearCloudFromOtherCloud(cloudSrc, cloudTgt, searchRadius);
  //showCloud2(cloudTgt, "No Background");
  pcl::io::savePLYFile(outFolderPath+"/CloudWithoutBackground.ply", *cloudWithoutBackground);
  //debug_showCombinedCloud(background, cloudTgt, "Background with cloudTgt");
  return 0;
}

enum JobName {
  Shapenet,
  Shapenet2,
  ShowCloudWithNormals,
  ComputeAndShowNormals,
  RegistrationFormat,
  IterativeScaleRegistration,
  BackgroundRemovalPipeline,
  UnknownJob
};

JobName jobStringToEnum(std::string jobString){
  if(jobString == "Shapenet") return Shapenet;
  if(jobString == "Shapenet2") return Shapenet2;
  if(jobString == "ShowCloudWithNormals") return ShowCloudWithNormals;
  if(jobString == "ComputeAndShowNormals") return ComputeAndShowNormals;
  if(jobString == "RegistrationFormat") return RegistrationFormat;
  if(jobString == "IterativeScaleRegistration") return IterativeScaleRegistration;
  if(jobString == "BackgroundRemovalPipeline") return BackgroundRemovalPipeline;
  return UnknownJob;
}

int main (int argc, char** argv)
{
  int target_thread_num = 4;
  omp_set_num_threads(target_thread_num);
  //testcgalRegistartion(argc, argv);
  //cgalMatchingExamplePointMatcher(argv[1], argv[2]);
  //cgalMatchingExampleOpenGR(argv[1], argv[2]);

  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("job,J", po::value<std::string>(), "Job that should be performed")
    ("in", po::value<std::string>(), "Path to ply file or folder that should be converted to Shapenet format")
    ("out", po::value<std::string>(), "Path to file or folder where converted Shapenet format should be saved")
    ("PointCloudName", po::value<std::string>(), "Name of the PointCloud that should be converted to Shapenet format")
    ("PointCloudPlant", po::value<std::string>(), "Path to Plant Point Cloud")
    ("NormalEsitmationSearchRadius", po::value<float>(), "Radius that should be used in Normal Estimation")
    ("SourceCloudPath", po::value<std::string>(), "Source cloud path for DCP Format transformation")
    ("TargetCloudPath", po::value<std::string>(), "Target cloud path for DCP Format transformation")
    ("OutputFolder", po::value<std::string>(), "Path to Folder where DCP Format should be saved")
    ("SubsamplePointCount", po::value<int>(), "Amount of Points that should be used for subsampling")
    ("RemoveBackground", po::value<bool>(), "Remove background when converting to shapnet format")
    ("MaxSubsample", po::value<int>(), "Maximum Subsample that should be created")
    ("SearchRadius", po::value<float>(), "Radius that should be used for nearest neighbor search")
    ("NoPlaneAlignment", po::bool_switch()->default_value(false), "Ignore plane alignment step")
    ("RotateRandom", po::bool_switch()->default_value(false), "Rotate cloud randomly")
    ("CenterOnly", po::bool_switch()->default_value(false), "Rotate cloud randomly")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);   

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  srand (time(NULL)); //set Random

  if (vm.count("job")) {
    cout << "Job " << vm["job"].as<std::string>() << " will be perfomed.\n";

    JobName jobName = jobStringToEnum(vm["job"].as<std::string>());

    switch(jobName){
      case Shapenet:
        convertToShapenetFormat(vm);
        break;
      case Shapenet2:
        return convertToShapenetFormat2(vm);
        break;
      case ShowCloudWithNormals:
        return showCloudWithNormals(vm);
      case ComputeAndShowNormals:
        return computeAndShowNormals(vm);
      case RegistrationFormat:
        return convertToRegistrationFormat(vm);
      case IterativeScaleRegistration:
        return iterativeScaleRegistration(vm);
      case BackgroundRemovalPipeline:
        return backgroundRemovalPipeline(vm);
      case UnknownJob:
        cout << "Job " << vm["job"].as<std::string>() << " is unknown.\n";
        return 1;
    }

    return 0;
  } else {
    cout << "You have to specify a Job that should be performed" << endl;
    return 1;
  }

  //testKeyPoints(argc, argv);
  /*std::map<std::string,std::string> testConfig = readConfig("../config/test.config");
  if(testConfig.find("registration_debug") != testConfig.end())
    cout << "config contains key registration_debug" << endl;*/

  //return backgroundRemovalPipeline(argc, argv);

}
