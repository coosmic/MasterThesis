#include <iostream>
#include <thread>

#include <stdio.h>
#include <stdlib.h>

#include <chrono>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/ascii_io.h>
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
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/common/io.h>

#include <pcl/filters/filter.h>

#include <pcl/features/principal_curvatures.h>

#include "vtkPlaneSource.h"

#include <omp.h>

#include <Eigen/Dense>


#define DEBUG true

#define PointType pcl::PointXYZRGBNormal

using namespace std::chrono_literals;

bool loadAsciCloud(std::string filename, pcl::PointCloud<PointType>::Ptr cloud)
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

        PointType point;
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

void showCloud2(pcl::PointCloud<PointType>::Ptr cloud, std::string windowName, pcl::ModelCoefficients::Ptr coefficientsPlane = nullptr, bool showNormals = false){

  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer (windowName));
  viewer->setBackgroundColor (0, 0, 0);

  pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgb(cloud);
  viewer->addPointCloud<PointType> (cloud, rgb, "sample cloud");
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

    //vtkSmartPointer<vtkPolyData> plane = createPlane(*coefficientsPlane,0,0,0,2.0);
    
    //viewer->addModelFromPolyData(plane,"myplane");

    viewer->addPlane (*coefficientsPlane, centroid[0],centroid[1],centroid[2], "plane");
    viewer->addPlane (coeffs, centroid[0],centroid[1],centroid[2], "planeBase");
  }

  if(showNormals){
    viewer->addPointCloudNormals<PointType, PointType> (cloud, cloud, 10, 0.15, "normals"); 

  }
  
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    std::this_thread::sleep_for(100ms);
  }

}

void stemSegmentation3(pcl::PointCloud<PointType>::Ptr cloud){
  
  pcl::SACSegmentationFromNormals<PointType, PointType> seg;
  pcl::ModelCoefficients::Ptr coefficients_cylinder (new pcl::ModelCoefficients);
  
  pcl::ExtractIndices<PointType> extract, extract2;

  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_CYLINDER);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setNormalDistanceWeight (0.1);
  seg.setMaxIterations (2000);
  seg.setDistanceThreshold (0.075);
  seg.setRadiusLimits (0, 0.15);
  seg.setInputCloud (cloud);
  seg.setInputNormals (cloud);

  pcl::PointCloud<PointType>::Ptr cloud_labeld (new pcl::PointCloud<PointType> ());
  copyPointCloud(*cloud, *cloud_labeld);

  pcl::PointCloud<PointType>::Ptr cloud_cylinder (new pcl::PointCloud<PointType> ());

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

void stemSegmentation2(pcl::PointCloud<PointType>::Ptr cloud, float searchRadius){
  pcl::PrincipalCurvaturesEstimation<PointType, PointType, pcl::PrincipalCurvatures> principalCurvaturesEstimation;
  principalCurvaturesEstimation.setInputCloud (cloud);
  principalCurvaturesEstimation.setInputNormals (cloud);

  pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType>);
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

void stemSegementation(pcl::PointCloud<PointType>::Ptr cloud, float searchRadius){

  pcl::KdTreeFLANN<PointType> kdtree;
  kdtree.setInputCloud (cloud);

  int pointsInCloud = cloud->points.size();

  //#pragma omp parallel num_threads(24)
  #pragma omp target teams distribute parallel for collapse(2)
  for(int i=0; i< pointsInCloud; i++){
    std::vector<int> neighborIndices; //to store index of surrounding points 
    std::vector<float> pointRadiusSquaredDistance; // to store distance to surrounding

    PointType searchPoint = cloud->points[i];
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

vtkSmartPointer<vtkPolyData> createPlane(const pcl::ModelCoefficients &coefficients, double x, double y, double z, double scale = 1.0)
{
	vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New();

	double norm_sqr = 1.0 / (coefficients.values[0] * coefficients.values[0] +
		coefficients.values[1] * coefficients.values[1] +
		coefficients.values[2] * coefficients.values[2]);

	plane->SetNormal(coefficients.values[0], coefficients.values[1], coefficients.values[2]);
	double t = x * coefficients.values[0] + y * coefficients.values[1] + z * coefficients.values[2] + coefficients.values[3];
	x -= coefficients.values[0] * t * norm_sqr;
	y -= coefficients.values[1] * t * norm_sqr;
	z -= coefficients.values[2] * t * norm_sqr;

	Eigen::Vector3d p1,p2;
	Eigen::Vector3d n;
	n.x() = x + coefficients.values[0];
	n.y() = y + coefficients.values[1];
	n.z() = z + coefficients.values[2];
	
	p1.x() = x+1.0;
	p1.y() = y;
	p1.z() = (coefficients.values[3] - coefficients.values[0]*p1.x() - coefficients.values[1]*p1.y())/coefficients.values[2];

	p2 = n.cross(p1);

	p1.normalize();
	p2.normalize();

	p1 = p1*scale;
	p2 = p2*scale;
	
	double point1[3];
	double point2[3];

	point1[0] = p1.x();
	point1[1] = p1.y();
	point1[2] = p1.z();

	point2[0] = p2.x();
	point2[1] = p2.y();
	point2[2] = p2.z();

	std::cout << "p1 = " << p1 << std::endl;
	std::cout << "p2 = " << p2 << std::endl;
	
	plane->SetOrigin(x, y, z);
	plane->SetPoint1(point1);
	plane->SetPoint2(point2);

	plane->Update();

	return (plane->GetOutput());
}


void rotateCloud(pcl::PointCloud<PointType>::Ptr cloud, pcl::ModelCoefficients::Ptr coefficientsPlane){

  Eigen::Matrix<float, 1, 3> floor_plane_normal_vector, xy_plane_normal_vector, rotation_vector;

  floor_plane_normal_vector[0] = coefficientsPlane->values[0];
  floor_plane_normal_vector[1] = coefficientsPlane->values[1];
  floor_plane_normal_vector[2] = coefficientsPlane->values[2];

  std::cout << floor_plane_normal_vector << std::endl;

  xy_plane_normal_vector[0] = 0.0;
  xy_plane_normal_vector[1] = 0.0;
  xy_plane_normal_vector[2] = 1.0;

  std::cout << xy_plane_normal_vector << std::endl;

  rotation_vector = xy_plane_normal_vector.cross (floor_plane_normal_vector);
  rotation_vector.normalize();
  std::cout << "Rotation Vector: "<< rotation_vector << std::endl;

  float theta = -acos(floor_plane_normal_vector.dot(xy_plane_normal_vector)/sqrt( pow(coefficientsPlane->values[0],2)+ pow(coefficientsPlane->values[1],2) + pow(coefficientsPlane->values[2],2)));


  Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();

  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*cloud, centroid);
    
    cout << "centroid: " << centroid[0] << " " <<  centroid[1] << " " <<   centroid[2] << " " <<   centroid[3] << " \n";

  transform_2.translation() << -centroid[0], -centroid[1], -centroid[2];
  transform_2.rotate (Eigen::AngleAxisf (theta, rotation_vector));
  std::cout << "Transformation matrix: " << std::endl << transform_2.matrix() << std::endl;

  pcl::transformPointCloud (*cloud, *cloud, transform_2);

  //showCloud2(cloud, "rotated PointCloud v2", coefficientsPlane);

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

void printMinMax(pcl::PointCloud<PointType>::Ptr cloud){

  PointType minPt, maxPt;
  pcl::getMinMax3D (*cloud, minPt, maxPt);
  std::cout << "Max x: " << maxPt.x << std::endl;
  std::cout << "Max y: " << maxPt.y << std::endl;
  std::cout << "Max z: " << maxPt.z << std::endl;
  std::cout << "Min x: " << minPt.x << std::endl;
  std::cout << "Min y: " << minPt.y << std::endl;
  std::cout << "Min z: " << minPt.z << std::endl;

}

void noiseFilter(pcl::PointCloud<PointType>::Ptr cloud, int minNumberNeighbors, float radius){
  pcl::RadiusOutlierRemoval<PointType> outrem;
  // build the filter
  outrem.setInputCloud(cloud);
  outrem.setRadiusSearch(radius);
  outrem.setMinNeighborsInRadius (minNumberNeighbors);
  outrem.setKeepOrganized(true);
  // apply filter
  outrem.filter (*cloud);

  pcl::PointIndices::Ptr nanIndices (new pcl::PointIndices);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
}

int main (int argc, char** argv)
{
  pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);

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

  

  //printMinMax(cloud);
  //showCloud(cloud, "PlainCloud");

  ///////////////////
  // Remove Planes //
  ///////////////////

  long MIN_POINTS_IN_PLANE = cloud->size()*0.4;
  cout << "min points in plane: "<<MIN_POINTS_IN_PLANE<<endl;

  long pointsInPlane = 0;

  //do{

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

    // Create the segmentation object
    pcl::SACSegmentation<PointType> seg;
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
      pcl::ExtractIndices<PointType> extract;
      extract.setInputCloud(cloud);
      extract.setIndices(inliers);
      extract.setNegative(true);
      extract.filter(*cloud);
    }
  
  //}
  //while(pointsInPlane > MIN_POINTS_IN_PLANE);

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
