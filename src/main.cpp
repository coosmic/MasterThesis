
#include "definitions.h"
#include "configuration.h"
#include "debug.h"
#include "registration_cgal.h"
#include "test.h"
#include "registration_pcl.h"
#include "utilities.h"
#include "utilities_io.h"
#include "gaussian.h"

#include <iostream>
#include <thread>
#include <unordered_set>

#include <stdio.h>
#include <stdlib.h>

#include <chrono>

#include <math.h>

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

bool debugShowResult = true;
bool debugShowStepResult = true;

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

  *cloud_labeld += *cloud;
  *cloud_labeld += *cloud_cylinder;


  showCloud2(cloud_labeld, "other");

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

      cloud->points[i].r = (uint8_t)((principalCurvatures->points[i].pc1 - 0.192002f) * pc1ScaleFactorToColor);
      cloud->points[i].g = 0;
      cloud->points[i].b = 0;
    }

    cout<< "maxpc1: "<<maxpc1 << "\nminpc1: "<<minpc1<<"\nmaxpc2: "<<maxpc2<<"\nminpc2: "<<minpc2<<endl;

    cout<< "maxpcx: "<<maxpcx << "\nminpcx: "<<minpcx<<"\nmaxpcy: "<<maxpcy<<"\nminpcy: "<<minpcy<<"\nmaxpcz: "<<maxpcz<<"\nminpcz: "<<minpcz<<endl;
    showCloud2(cloud, "intensity");
  }

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

int testStemSegmentation(std::string pathToFile, int classifierNumber, po::variables_map vm){
  pcl::PointCloud<PointTypePCL>::Ptr cloud(new pcl::PointCloud<PointTypePCL>);

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
    showCloud2(cloud, "raw asci cloud");
      
  } else{
    PCL_ERROR ("Unkown cloud type %s \n", pathToFile);
    return -1;
  }

  bool doNoiseFilter = true;
  int noiseFilterMinNeighbors1 = 30;
  int noiseFilterMinNeighbors2 = 500;
  int noiseFilterMinNeighbors3 = 1000;
  float noiseFilterRadius1 = 0.8;
  float noiseFilterRadius2 = 3.0;
  float noiseFilterRadius3 = 10.0;

  if(vm.count("NoiseFilterActive")){
    doNoiseFilter = vm["NoiseFilterActive"].as<bool>();
    if(doNoiseFilter){
      if(vm.count("NoiseFilterMinNeighbors1"))
        noiseFilterMinNeighbors1 = vm["NoiseFilterMinNeighbors1"].as<int>();
      if(vm.count("NoiseFilterMinNeighbors2"))
        noiseFilterMinNeighbors2 = vm["NoiseFilterMinNeighbors2"].as<int>();
      if(vm.count("NoiseFilterMinNeighbors3"))
        noiseFilterMinNeighbors3 = vm["NoiseFilterMinNeighbors3"].as<int>();
      if(vm.count("NoiseFilterRadius1"))
        noiseFilterRadius1 = vm["NoiseFilterRadius1"].as<float>();
      if(vm.count("NoiseFilterRadius2"))
        noiseFilterRadius2 = vm["NoiseFilterRadius2"].as<float>();
      if(vm.count("NoiseFilterRadius3"))
        noiseFilterRadius3 = vm["NoiseFilterRadius3"].as<float>();
    }
  }

  centerCloud(cloud);

  ///////////////////
  // Remove Planes //
  ///////////////////

  pcl::ModelCoefficients::Ptr coefficients = planeFilter(cloud);

  if(doNoiseFilter){
    //////////////////////////
    // Remove Noise Level 1 //
    //////////////////////////
    if(noiseFilterMinNeighbors1 > 0)
      noiseFilter(cloud, noiseFilterMinNeighbors1, noiseFilterRadius1);

    //showCloud(cloud, "Noise1");

    //////////////////////////
    // Remove Noise Level 2 //
    //////////////////////////
    if(noiseFilterMinNeighbors2 > 0)
      noiseFilter(cloud, noiseFilterMinNeighbors2, noiseFilterRadius2);

    //////////////////////////
    // Remove Noise Level 3 //
    //////////////////////////
    if(noiseFilterMinNeighbors3 > 0)
      noiseFilter(cloud, noiseFilterMinNeighbors3, noiseFilterRadius3);
  }
  
  rotateCloud(cloud, coefficients);

  printMinMax(cloud);

  float searchRadius;
  if(vm["CalculateNormals"].as<bool>()){
    if(!vm.count("SearchRadius")){
      cout << "Missing parameter SearchRadius\n";
      return 1;
    }
    searchRadius = vm["SearchRadius"].as<float>();
    pcl::NormalEstimation<PointTypePCL, PointTypePCL> ne;
    pcl::PointCloud<PointTypePCL>::Ptr src_normals_ptr (new pcl::PointCloud<PointTypePCL>);
    pcl::PointCloud<PointTypePCL>& src_normals = *src_normals_ptr;
    pcl::search::KdTree<PointTypePCL>::Ptr tree_xyz (new pcl::search::KdTree<PointTypePCL>());
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree_xyz);
    ne.setRadiusSearch(searchRadius);
    ne.compute(*src_normals_ptr);
    for(size_t i = 0;  i < src_normals.points.size(); ++i) {
        src_normals.points[i].x = cloud->points[i].x;
        src_normals.points[i].y = cloud->points[i].y;
        src_normals.points[i].z = cloud->points[i].z;
    }
    if(debugShowStepResult)showCloud2(src_normals_ptr, "Cloud with normals", nullptr, true);
    cloud = src_normals_ptr;
  }

  
  switch(classifierNumber){
    case 0:
      if(!vm.count("SearchRadius")){
        cout << "Missing parameter SearchRadius\n";
        return 1;
      }
      searchRadius = vm["SearchRadius"].as<float>();
      stemSegementation(cloud, searchRadius);
      break;
    case 1:
      if(!vm.count("SearchRadius")){
        cout << "Missing parameter SearchRadius\n";
        return 1;
      }
      searchRadius = vm["SearchRadius"].as<float>();
      stemSegmentation2(cloud, searchRadius);
      break;
    case 2:
      stemSegmentation3(cloud);
      break;
    default:
      return 1;
  }

  return (0);
}



void matchClouds(pcl::PointCloud<PointTypePCL>::Ptr cloudA, pcl::PointCloud<PointTypePCL>::Ptr cloudB, po::variables_map vm){

  bool doNoiseFilter = true;
  int noiseFilterMinNeighbors1 = 50;
  int noiseFilterMinNeighbors2 = 1400;
  float noiseFilterRadius1 = 0.08;
  float noiseFilterRadius2 = 0.2;

  if(vm.count("NoiseFilterActive")){
    doNoiseFilter = vm["NoiseFilterActive"].as<bool>();
    if(doNoiseFilter){
      if(vm.count("NoiseFilterMinNeighbors1"))
        noiseFilterMinNeighbors1 = vm["NoiseFilterMinNeighbors1"].as<int>();
      if(vm.count("NoiseFilterMinNeighbors2"))
        noiseFilterMinNeighbors2 = vm["NoiseFilterMinNeighbors2"].as<int>();
      if(vm.count("NoiseFilterRadius1"))
        noiseFilterRadius1 = vm["NoiseFilterRadius1"].as<float>();
      if(vm.count("NoiseFilterRadius2"))
        noiseFilterRadius2 = vm["NoiseFilterRadius2"].as<float>();
    }
  }

  float voxelSize=0.015;
  if(vm.count("VoxelSize"))
    voxelSize = vm["VoxelSize"].as<float>();

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
  if(debugShowStepResult)debug_showCombinedCloud(cloudA, cloudB, "rescaled src Cloud");

  cout << "downsampling clouds..." <<endl;
  downsampleCloud(cloudA, voxelSize);
  downsampleCloud(cloudB, voxelSize);
  cout << "cloudA Points: "<<cloudA->size()<<endl; 
  //debug_showCombinedCloud(cloudA, cloudB, "downsampled Clouds");

  if(doNoiseFilter){
    cout << "noise filtering clouds..." <<endl;
    noiseFilter(cloudA, noiseFilterMinNeighbors1, noiseFilterRadius1, noiseFilterMinNeighbors2, noiseFilterRadius2);
    noiseFilter(cloudB, noiseFilterMinNeighbors1, noiseFilterRadius1, noiseFilterMinNeighbors2, noiseFilterRadius2);
    if(debugShowStepResult)debug_showCombinedCloud(cloudA, cloudB, "noise filtered Clouds");
  }
  
  registrationPipeline(cloudA, cloudB, true, true, true, false, true); 

  cout << "Transform clouds to same size..." <<endl;
  transformCloudsToSameSize(cloudA, cloudB);
  centerCloud(cloudA);
  if(debugShowStepResult)debug_showCombinedCloud(cloudA, cloudB, "rescaled src Cloud");
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

int manuellRegistrationPipeline(po::variables_map vm){

  if(!vm.count("SourceCloudPath")){
    cout << "Missing Parameter SourceCloudPath\n";
    return 1;
  }

  if(!vm.count("TargetCloudPath")){
    cout << "Missing Parameter TargetCloudPath\n";
    return 1;
  }

  bool doSegmentation = false;
  if(vm.count("SegmentationAfterRegistration"))
    doSegmentation = vm["SegmentationAfterRegistration"].as<bool>();

  std::string pathToBackground = vm["TargetCloudPath"].as<std::string>();
  std::string pathToPlant = vm["SourceCloudPath"].as<std::string>();

  pcl::PointCloud<PointTypePCL>::Ptr cloudPlant(new pcl::PointCloud<PointTypePCL>);
  pcl::PointCloud<PointTypePCL>::Ptr cloudBackground(new pcl::PointCloud<PointTypePCL>);

  if( pcl::io::loadPLYFile(pathToPlant, *cloudPlant) == -1){

    PCL_ERROR ("Couldn't read ply file %s \n", pathToPlant);
    return (-1);

  }
  if( pcl::io::loadPLYFile(pathToBackground, *cloudBackground) == -1){

    PCL_ERROR ("Couldn't read ply %s file\n", pathToBackground);
    return (-1);

  }

  backgroundPreparation(cloudBackground);

  setColorChannelExclusive(cloudPlant, ColorChannel::b, 255);
  setColorChannelExclusive(cloudBackground, ColorChannel::r, 255);

  //debug_showCombinedCloud(cloudPlant, cloudBackground, "ChannelSwap");

  matchClouds(cloudPlant, cloudBackground, vm);
  if(debugShowResult)debug_showCombinedCloud(cloudBackground, cloudPlant, "Matched Clouds");

  if(doSegmentation){
    substractCloudFromOtherCloud(cloudBackground, cloudPlant, 0.05);

    stemSegmentation2(cloudPlant, 3.0);

    if(debugShowResult)showCloud2(cloudPlant, "Classified Cloud");
  }
  
  //t.join();
  return 0;

}

int handcraftedStemSegmentation(po::variables_map vm){

  if(!vm.count("in")){
    cout << "Missing parameter in\n";
    return 1;
  }
  std::string pathIn = vm["in"].as<std::string>();
  if(!vm.count("Classifier")){
    cout << "Missing parameter in\n";
    return 1;
  }
  int classifierNumber = vm["Classifier"].as<int>();
  if(classifierNumber < 0 || classifierNumber > 2){
    cout << "Unsuported Classifier "<< classifierNumber <<"\n";
    return 1;
  }

  testStemSegmentation(pathIn, classifierNumber, vm);

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

  pcl::PointCloud<PointTypePCL>::Ptr cloudSrc = loadAnyCloud(srcCloudPath);

  pcl::PointCloud<PointTypePCL>::Ptr cloudTgt = loadAnyCloud(tgtCloudPath);

  pcl::ModelCoefficients::Ptr coefficientsA (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliersA (new pcl::PointIndices);
  findPlaneInCloud(cloudSrc, coefficientsA, inliersA);
  rotateCloud(cloudSrc, coefficientsA);
  //removePointsInCloud(cloudSrc, inliersA);
  findPlaneInCloud(cloudTgt, coefficientsA, inliersA);
  rotateCloud(cloudTgt, coefficientsA);
  //removePointsInCloud(cloudTgt, inliersA);

  bool shoudlExtractCenterOfCloud = true;
  if(vm.count("CenterOnly")){
    shoudlExtractCenterOfCloud = vm["CenterOnly"].as<bool>();
  }

  if(shoudlExtractCenterOfCloud){
    cout << "Extracting Center of Clouds\n";
    extractCenterOfCloud(cloudSrc, 0.3);
    extractCenterOfCloud(cloudTgt, 0.3);
  }
  
  Cloud::Ptr backgroundSrc = removeBackgroundPointsShapenet(cloudSrc);
  //showCloud2(backgroundSrc, "Background Src");

  setColorChannelExclusive(backgroundSrc, ColorChannel::r, 255);
  setColorChannelExclusive(backgroundSrc, ColorChannel::g, 0);
  setColorChannelExclusive(backgroundSrc, ColorChannel::b, 0);

  setColorChannelExclusive(cloudTgt, ColorChannel::r, 0);
  setColorChannelExclusive(cloudTgt, ColorChannel::g, 255);
  setColorChannelExclusive(cloudTgt, ColorChannel::b, 0);

  pcl::PointCloud<PointTypePCL>::Ptr cloudSrcSubsampled = subSampleCloudRandom(backgroundSrc, subsampleCount);
  pcl::PointCloud<PointTypePCL>::Ptr cloudTgtSubsampled = subSampleCloudRandom(cloudTgt, subsampleCount);
  

  bool shouldTransformToShapenet = true;
  if(vm.count("UseShapenetFormat")){
    shouldTransformToShapenet = vm["UseShapenetFormat"].as<bool>();
  }
  Eigen::Matrix4f ts,ss,tt,st;
  if(shouldTransformToShapenet){
    transformToShapenetFormat(cloudTgtSubsampled, tt,st);
    transformToShapenetFormat(cloudSrcSubsampled, ts,ss); 
    //debug_showCombinedCloud(cloudSrcSubsampled, cloudTgtSubsampled, "ShapenetFormat");
  } else
    ts = ss = tt = st = Eigen::Matrix4f::Identity();

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

  std::cout << "Best Scale: " << (scaleFactor * (float)bestScaleStep)<<std::endl;

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

  bool extractCenter = true;
  if(vm.count("CenterOnly")){
    extractCenter = vm["CenterOnly"].as<bool>();
  }
  if(extractCenter)
    extractCenterOfCloud(cloudSrc, 0.3);

  int numOfPoints = 1024;
  if(vm.count("SubsamplePointCount")){
    numOfPoints = vm["SubsamplePointCount"].as<int>();
  }

  pcl::PointCloud<PointTypePCL>::Ptr cloudSrcSubsampled = subSampleCloudRandom(cloudSrc, numOfPoints);

  if(vm["UseShapenetFormat"].as<bool>()){
    Eigen::Matrix4f t,s;
    transformToShapenetFormat(cloudSrcSubsampled, t,s);
  }
  

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

int surfaceGenerator(po::variables_map vm){
  
  if(!vm.count("PointCountX")){
      cout << "Missing parameter PointCountX\n";
      return 1;
  }

  if(!vm.count("PointCountY")){
      cout << "Missing parameter PointCountY\n";
      return 1;
  }

  if(!vm.count("NumberOfWaves")){
      cout << "Missing parameter NumberOfWaves\n";
      return 1;
  }

  if(!vm.count("WaveAmplitude")){
      cout << "Missing parameter WaveAmplitude\n";
      return 1;
  }

  int width = vm["PointCountX"].as<int>();
  int height = vm["PointCountY"].as<int>();
  float numberOfWaves = vm["NumberOfWaves"].as<float>();
  float amplitude = vm["WaveAmplitude"].as<float>();
  float widthFaktor = 1.f/width;
  float heightFaktor = 1.f/height;
  float minDistanceBetweenPoints = min(widthFaktor, heightFaktor);
  float maxDistanceBetweenPoints = max(widthFaktor, heightFaktor);
  
  bool jitter = true;
  if(vm.count("Jitter")){
    jitter = vm["Jitter"].as<bool>();
  }
  float maxJitter;
  float jitterScale = 1.0;
  if(jitter){
    if(vm.count("JitterScale")){
      jitterScale = vm["JitterScale"].as<float>();
    }
    maxJitter = minDistanceBetweenPoints * jitterScale;
  }

  bool slope = false;
  if(vm.count("Slope")){
    slope = vm["Slope"].as<bool>();
  }
  float slopeAmplitued = 1.0;
  if(slope){
    cout << "Slope active\n";
    if(vm.count("SlopeAmplitude")){
      slopeAmplitued = vm["SlopeAmplitude"].as<float>();
    }
  }

  bool gaussian = false;
  if(vm.count("Gaussian")){
    gaussian = vm["Gaussian"].as<bool>();
  }
  float sigma = 0.5;
  if(gaussian){
    cout << "Gaussian active\n";
    if(vm.count("Sigma")){
      sigma = vm["Sigma"].as<float>();
    }
  }
  Matrix gaussKernel = getGaussianKernel(height,width, sigma, sigma);

  pcl::PointCloud<PointTypePCL> cloud;
  cloud.width = width;
  cloud.height = height;
  cloud.is_dense = false;
  cloud.resize (width * height);

  float sinusVal = 0.0;
  float iterValue = 0.0;
  float stepWidth = (2 * M_PI) * numberOfWaves * widthFaktor;
  float maxZJitter = amplitude * sin(stepWidth) * jitterScale;
  float offset = 0.0;
  float finalOffset = 0.0;
  for(int i=0; i<width; ++i){
    if(slope)
      offset = slopeAmplitued * sin(i*widthFaktor);
    for(int j=0; j<height; ++j){
      if(gaussian)
        finalOffset = offset + gaussKernel[i][j];
      else
        finalOffset = offset;
      int index = i*width + j;
      if(jitter){
        cloud.points[index].x = (i * widthFaktor) + (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * maxJitter;
        cloud.points[index].y = (j * heightFaktor) + (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * maxJitter;
        cloud.points[index].z = iterValue + finalOffset + ((static_cast <float> (rand()) / static_cast <float> (RAND_MAX)) * maxZJitter);
      } else {
        cloud.points[index].x =   i * widthFaktor;
        cloud.points[index].y =   j * heightFaktor;
        cloud.points[index].z =   iterValue + finalOffset;
      }

      cloud.points[index].r = 255;
      cloud.points[index].g = 0;
      cloud.points[index].b = 0;
    }
    sinusVal += stepWidth;
    iterValue = amplitude*sin(sinusVal);
  }

  pcl::PointCloud<PointTypePCL>::Ptr cloud_ptr = cloud.makeShared();

  if(vm["CalculateNormals"].as<bool>()){
    float searchRadius = maxDistanceBetweenPoints*3;
    cloud_ptr = calculateNormals(cloud_ptr, searchRadius);
  }

  for(int i=0; i<cloud_ptr->size(); ++i){
    if(cloud_ptr->points[i].normal_z < 0){
      cloud_ptr->points[i].normal_x = -cloud_ptr->points[i].normal_x;
      cloud_ptr->points[i].normal_y = -cloud_ptr->points[i].normal_y;
      cloud_ptr->points[i].normal_z = -cloud_ptr->points[i].normal_z;
    }
  }

  if(debugShowResult)showCloud2(cloud_ptr, "Generated Background", nullptr, true);

  if(vm.count("out")){
    pcl::io::savePLYFileBinary(vm["out"].as<std::string>(), *cloud_ptr);
  }
  
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
  HandcraftedStemSegmentation,
  ManuellRegistrationPipeline,
  SurfaceGenerator,
  UnknownJob
};

JobName jobStringToEnum(std::string jobString){
  if(jobString == "Shapenet") return Shapenet;
  if(jobString == "Shapenet2") return Shapenet2;
  if(jobString == "ShowCloudWithNormals") return ShowCloudWithNormals;
  if(jobString == "ComputeAndShowNormals") return ComputeAndShowNormals;
  if(jobString == "RegistrationFormat") return RegistrationFormat;
  if(jobString == "IterativeScaleRegistration") return IterativeScaleRegistration;
  if(jobString == "ManuellRegistrationPipeline") return ManuellRegistrationPipeline;
  if(jobString == "HandcraftedStemSegmentation") return HandcraftedStemSegmentation;
  if(jobString == "SurfaceGenerator") return SurfaceGenerator;
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
    ("CenterOnly", po::value<bool>()->default_value(false), "Extrect center of cloud")
    ("DebugShowResults", po::bool_switch()->default_value(true), "Show Results of Jobs if available")
    ("DebugShowStepResults", po::bool_switch()->default_value(true), "Show Step Results of Jobs if available")
    ("NoiseFilterActive", po::bool_switch()->default_value(false), "Noise filter clouds")
    ("NoiseFilterMinNeighbors1", po::value<int>(), "Number of Neighbors that should be in the neighborhood of each point (0 or negativ number if level should be skipped)")
    ("NoiseFilterRadius1", po::value<float>(), "Radius that should be used for noise filter")
    ("NoiseFilterMinNeighbors2", po::value<int>(), "Number of Neighbors that should be in the neighborhood of each point (0 or negativ number if level should be skipped)")
    ("NoiseFilterRadius2", po::value<float>(), "Radius that should be used for noise filter")
    ("NoiseFilterMinNeighbors3", po::value<int>(), "Number of Neighbors that should be in the neighborhood of each point (0 or negativ number if level should be skipped)")
    ("NoiseFilterRadius3", po::value<float>(), "Radius that should be used for noise filter")
    ("VoxelSize", po::value<float>(), "Used for Downsampling")
    ("SegmentationAfterRegistration", po::bool_switch()->default_value(false), "Should target be substracted from source followed by segmentation of Source?")
    ("Classifier", po::value<int>()->default_value(1), "Classifier that should be used for HandcraftedStemSegmentation. 0: 1: 2:")
    ("CalculateNormals", po::value<bool>()->default_value(true), "CalculateNormals")
    ("PointCountX", po::value<int>(), "Number of Points in x Direction")
    ("PointCountY", po::value<int>(), "Number of Points in y Direction")
    ("NumberOfWaves", po::value<float>(), "Number of Waves that should be generated")
    ("WaveAmplitude", po::value<float>(), "Amplitude of the Waves")
    ("JitterScale", po::value<float>(), "minDistanceBetweenPoints * JitterScale = maxJitterDistance")
    ("Jitter", po::value<bool>()->default_value(true), "Should generated points be Jittered?")
    ("Slope", po::value<bool>()->default_value(false), "Should Surface be with Slope?")
    ("SlopeAmplitude", po::value<float>(), "Amplitude of the Sinus used for Slope")
    ("Gaussian", po::value<bool>()->default_value(false), "Apply Gaussian Distribution on surface")
    ("Sigma", po::value<float>(), "Sigma for Gauss Distribution")
    ("UseShapenetFormat", po::value<bool>()->default_value(true), "Use Shapent Format in iterative scale registration")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);   

  if (vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  if(vm.count("DebugShowResults"))
    debugShowResult = vm["DebugShowResults"].as<bool>();

  if(vm.count("DebugShowStepResults"))
    debugShowStepResult = vm["DebugShowStepResults"].as<bool>();

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
      case ManuellRegistrationPipeline:
        return manuellRegistrationPipeline(vm);
      case HandcraftedStemSegmentation:
        return handcraftedStemSegmentation(vm);
      case SurfaceGenerator:
        return surfaceGenerator(vm);
      case UnknownJob:
        cout << "Job " << vm["job"].as<std::string>() << " is unknown.\n";
        return 1;
    }

    return 0;
  } else {
    cout << "You have to specify a Job that should be performed" << endl;
    return 1;
  }
}
