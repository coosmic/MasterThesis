
#include "definitions.h"
#include "configuration.h"
#include "debug.h"
#include "registration_cgal.h"
#include "test.h"
#include "registration_pcl.h"
#include "utilities.h"


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


#define DEBUG true

//#define PointTypePCL pcl::PointXYZRGBNormal

using namespace std::chrono_literals;

bool loadAsciCloud(std::string filename, pcl::PointCloud<PointTypePCL>::Ptr cloud)
{
    std::cout << "Begin Loading Model" << std::endl;
    FILE* f = fopen(filename.c_str(), "r");

    if (NULL == f)
    {
        std::cout << "ERROR: failed to open file: " << filename << endl;
        return false;
    }

    float x, y, z;
    char r, g, b;
    float x_n, y_n, z_n;

    while (!feof(f))
    {
        int n_args = fscanf(f, "%f %f %f %c %c %c %f %f %f", &x, &y, &z, &r, &g, &b, &x_n, &y_n, &z_n);
        if (n_args != 9)
            continue;

        PointTypePCL point;
        point.x = x; 
        point.y = y; 
        point.z = z;
        point.r = r;
        point.g = g;
        point.b = b;
        point.normal_x = x_n;
        point.normal_y = y_n;
        point.normal_z = z_n;

        cloud->push_back(point);
    }

    fclose(f);

    std::cout << "Loaded cloud with " << cloud->size() << " points." << std::endl;

    return cloud->size() > 0;
}



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

void stemSegmentation2(pcl::PointCloud<PointTypePCL>::Ptr cloud, float searchRadius){
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

  if(false){
    float minpc1 = FLT_MAX, maxpc1 = FLT_MIN, minpc2 = FLT_MAX, maxpc2 = FLT_MIN;
    for(int i=0; i<principalCurvatures->points.size(); ++i){
      if(principalCurvatures->points[i].pc1 < minpc1)
        minpc1 = principalCurvatures->points[i].pc1;
      else if(principalCurvatures->points[i].pc1 > maxpc1)
        maxpc1 = principalCurvatures->points[i].pc1;

      if(principalCurvatures->points[i].pc2 < minpc2)
        minpc2 = principalCurvatures->points[i].pc2;
      else if(principalCurvatures->points[i].pc1 > maxpc2)
        maxpc2 = principalCurvatures->points[i].pc2;
      //std::cout << i << " pc1: "<< principalCurvatures->points[i].pc1 << " pc2: " << principalCurvatures->points[i].pc2 << std::endl;
    }

    cout<< "maxpc1: "<<maxpc1 << "\nminpc1: "<<minpc1<<"\nmaxpc2: "<<maxpc2<<"\nminpc2: "<<minpc2<<endl;

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

    for(int i=0; i<principalCurvatures->points.size(); ++i){
      float linearity_i = (principalCurvatures->points[i].pc1 - principalCurvatures->points[i].pc2) / principalCurvatures->points[i].pc1;

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
  #pragma omp target teams distribute parallel for collapse(2)
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

  pcl::ModelCoefficients::Ptr coefficientsB (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliersB(new pcl::PointIndices);
  findPlaneInCloud(cloudB, coefficientsB, inliersB);

  rotateCloud(cloudA, coefficientsA);
  rotateCloud(cloudB, coefficientsB);

  //debug_showCombinedCloud(cloudA, cloudB, "rotated Clouds");
  cout << "Transform clouds to same size..." <<endl;
  transformCloudsToSameSize(cloudA, cloudB);
  //debug_showCombinedCloud(cloudA, cloudB, "prescaled Clouds");

  cout << "downsampling clouds..." <<endl;
  downsampleCloud(cloudA, 0.05f);
  downsampleCloud(cloudB, 0.05f);
  //debug_showCombinedCloud(cloudA, cloudB, "downsampled Clouds");

  cout << "noise filtering clouds..." <<endl;
  noiseFilter(cloudA);
  noiseFilter(cloudB);
  //debug_showCombinedCloud(cloudA, cloudB, "noise filtered Clouds");

  cout << "extracting center of clouds..." <<endl;
  extractCenterOfCloud(cloudA, 0.3);
  extractCenterOfCloud(cloudB, 0.3);
  //debug_showCombinedCloud(cloudA, cloudB, "centered Clouds");

  cout << "register clouds feature based..." <<endl;

  if(registration_recalculateNormals){
    pcl::PointCloud<PointTypeRegistration>::Ptr src(new pcl::PointCloud<PointTypeRegistration>);
     pcl::PointCloud<pcl::PointNormal>::Ptr src_normals(new pcl::PointCloud<pcl::PointNormal>);
    copyPointCloud(*cloudA, *src);
    //copyPointCloud(*cloudA, *src_normals);
    pcl::PointCloud<PointTypeRegistration>::Ptr target(new pcl::PointCloud<PointTypeRegistration>);
    pcl::PointCloud<pcl::PointNormal>::Ptr target_normals(new pcl::PointCloud<pcl::PointNormal>);
    copyPointCloud(*cloudB, *target);
    //copyPointCloud(*cloudB, *target_normals);
    target_normals->resize(cloudB->size());
    src_normals->resize(cloudA->size());

    //showCloud2(target, "Target Normals", target_normals);

    Eigen::Matrix4f initialTransformation = registerClouds(src, target, true);

    debug_showCombinedCloud(src, target, "Feature Registration");

    cout << "register clouds with icp and estimate scale..." <<endl;
    pcl::IterativeClosestPoint<PointTypeRegistration, PointTypeRegistration> icp2;
    icp2.setInputSource(src);
    icp2.setInputTarget(target);

    boost::shared_ptr<pcl::registration::TransformationEstimationSVDScale <PointTypeRegistration, PointTypeRegistration>> teSVDscale (new pcl::registration::TransformationEstimationSVDScale <PointTypeRegistration, PointTypeRegistration>());
    icp2.setTransformationEstimation (teSVDscale);

    pcl::PointCloud<PointTypeRegistration> Unused2;
    icp2.align(Unused2);

    Eigen::Matrix4f icpTransformation = icp2.getFinalTransformation();
    pcl::transformPointCloud (*src, *src, icpTransformation);

    Eigen::Matrix4f finalTransformation = icpTransformation * initialTransformation;
    pcl::transformPointCloud (*cloudA, *cloudA, finalTransformation);
  
  } else {

    Eigen::Matrix4f initialTransformation = registerClouds(cloudA, cloudB, true);
    debug_showCombinedCloud(cloudA, cloudB, "Feature Registration");

    cout << "register clouds with icp and estimate scale..." <<endl;
    pcl::IterativeClosestPoint<PointTypePCL, PointTypePCL> icp2;
    icp2.setInputSource(cloudA);
    icp2.setInputTarget(cloudB);

    boost::shared_ptr<pcl::registration::TransformationEstimationSVDScale <PointTypePCL, PointTypePCL>> teSVDscale (new pcl::registration::TransformationEstimationSVDScale <PointTypePCL, PointTypePCL>());
    icp2.setTransformationEstimation (teSVDscale);

    pcl::PointCloud<PointTypePCL> Unused2;
    icp2.align(Unused2);

    Eigen::Matrix4f icpTransformation = icp2.getFinalTransformation();
    pcl::transformPointCloud (*cloudA, *cloudA, icpTransformation);
  }
  debug_showCombinedCloud(cloudA, cloudB, "scale alligned Clouds");

  //Eigen::Matrix4f finalTransformation = icpTransformation * initialTransformation;
  //pcl::transformPointCloud (*cloudA, *cloudA, finalTransformation);

  debug_showCombinedCloud(cloudA, cloudB, "Scale Transformation");

  cout << "register clouds with icp and remove plane..." <<endl;
  //remove planes
  planeFilter(cloudA);
  planeFilter(cloudB);
  
  pcl::IterativeClosestPoint<PointTypePCL, PointTypePCL> icp;
  icp.setInputSource(cloudA);
  icp.setInputTarget(cloudB);

  pcl::PointCloud<PointTypePCL> Unused;
  icp.align(Unused);

  pcl::transformPointCloud (*cloudA, *cloudA, icp.getFinalTransformation());

  //debug_showCombinedCloud(cloudA, cloudB, "simple alligned Clouds");

}

void switchRBColorChannel(Cloud::Ptr cloud){
  #pragma omp parallel for
  for(int i=0; i< cloud->points.size(); ++i){
    uint8_t tmpR = cloud->points[i].r;

    cloud->points[i].r = cloud->points[i].b;
    cloud->points[i].b = tmpR;
  }
}

int backgroundRemovalPipeline(int argc, char** argv){

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

  //switchRBColorChannel(cloudPlant);
  //switchRBColorChannel(cloudBackground);

  matchClouds(cloudBackground, cloudPlant);
 
  substractCloudFromOtherCloud(cloudBackground, cloudPlant);

  showCloud2(cloudPlant, "Cloud without Background");

  return 0;

}

int main (int argc, char** argv)
{
  //testcgalRegistartion(argc, argv);
  //cgalMatchingExamplePointMatcher(argv[1], argv[2]);
  //cgalMatchingExampleOpenGR(argv[1], argv[2]);

  //testKeyPoints(argc, argv);

  return backgroundRemovalPipeline(argc, argv);
}
