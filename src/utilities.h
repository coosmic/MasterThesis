#pragma once

#include "definitions.h"

#include "vtkPlaneSource.h"

#include <pcl/ModelCoefficients.h>

#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/common/pca.h>

#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/filters/filter.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/extract_indices.h>

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


void rotateCloud(pcl::PointCloud<PointTypePCL>::Ptr cloud, pcl::ModelCoefficients::Ptr coefficientsPlane){

  Eigen::Matrix<float, 1, 3> floor_plane_normal_vector, xy_plane_normal_vector, rotation_vector;

  floor_plane_normal_vector[0] = coefficientsPlane->values[0];
  floor_plane_normal_vector[1] = coefficientsPlane->values[1];
  floor_plane_normal_vector[2] = coefficientsPlane->values[2];

  //std::cout << floor_plane_normal_vector << std::endl;

  xy_plane_normal_vector[0] = 0.0;
  xy_plane_normal_vector[1] = 0.0;
  xy_plane_normal_vector[2] = 1.0;

  //std::cout << xy_plane_normal_vector << std::endl;

  rotation_vector = xy_plane_normal_vector.cross (floor_plane_normal_vector);
  rotation_vector.normalize();
  //std::cout << "Rotation Vector: "<< rotation_vector << std::endl;

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

void noiseFilter(pcl::PointCloud<PointTypePCL>::Ptr cloud, int minNumberNeighbors, float radius){
  pcl::RadiusOutlierRemoval<PointTypePCL> outrem;
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

void noiseFilter(pcl::PointCloud<PointTypePCL>::Ptr cloud){
  //////////////////////////
  // Remove Noise Level 1 //
  //////////////////////////
  
  noiseFilter(cloud, 30, 0.8);

  //////////////////////////
  // Remove Noise Level 2 //
  //////////////////////////

  noiseFilter(cloud, 1000, 2.0);
}

void findPlaneInCloud (pcl::PointCloud<PointTypePCL>::Ptr cloud, pcl::ModelCoefficients::Ptr coefficients, pcl::PointIndices::Ptr inliers){
  // Create the segmentation object
  pcl::SACSegmentation<PointTypePCL> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.5);

  seg.setInputCloud (cloud);
  seg.segment (*inliers, *coefficients);
}

void removePointsInCloud(pcl::PointCloud<PointTypePCL>::Ptr cloud, pcl::PointIndices::Ptr inliers){
  pcl::ExtractIndices<PointTypePCL> extract;
  extract.setInputCloud(cloud);
  extract.setIndices(inliers);
  extract.setNegative(true);
  extract.filter(*cloud);
}

pcl::ModelCoefficients::Ptr planeFilter(pcl::PointCloud<PointTypePCL>::Ptr cloud){
  long MIN_POINTS_IN_PLANE = cloud->size()*0.4;
  cout << "min points in plane: "<<MIN_POINTS_IN_PLANE<<endl;

  long pointsInPlane = 0;

  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

  findPlaneInCloud(cloud, coefficients, inliers);

  pointsInPlane = inliers->indices.size();

  if(pointsInPlane > MIN_POINTS_IN_PLANE){
    removePointsInCloud(cloud, inliers);
  }

  return coefficients;
}


/**
 * Expects a centered cloud as input.
 */
void extractCenterOfCloud(pcl::PointCloud<PointTypePCL>::Ptr cloud, double centerPortion){
  PointTypePCL min, max;
  pcl::getMinMax3D(*cloud, min, max);

  double xLength, yLength, zLength;
  xLength = min.x < 0.0 ? abs(min.x) + abs(max.x) : abs(max.x) - abs(min.x);
  yLength = min.y < 0.0 ? abs(min.y) + abs(max.y) : abs(max.y) - abs(min.y);
  //zLength = min.z < 0.0 ? abs(min.z) + abs(max.z) : abs(max.z) - abs(min.z);

  double searchRadius;
  if(xLength > yLength){
    searchRadius = yLength * centerPortion;
  } else {
    searchRadius = xLength * centerPortion;
  }

  pcl::KdTreeFLANN<PointTypePCL> kdtree;
  kdtree.setInputCloud (cloud);


  pcl::PCA<PointTypePCL> pca;
  pca.setInputCloud(cloud);
  Eigen::Vector4f meanVector = pca.getMean();

  PointTypePCL searchPoint;
  searchPoint.x = meanVector.x();
  searchPoint.y = meanVector.y();
  searchPoint.z = meanVector.z();

  std::vector<int> centerIndices;
  std::vector<float> pointRadiusSquaredDistance; // to store distance to surrounding
  kdtree.radiusSearch(searchPoint, searchRadius, centerIndices, pointRadiusSquaredDistance);
  while(searchPoint.z+searchRadius < max.z){
    searchPoint.z+=searchRadius;
    std::vector<int> neighborIndices; //to store index of surrounding points 
    
    kdtree.radiusSearch(searchPoint, searchRadius, neighborIndices, pointRadiusSquaredDistance);
    centerIndices.insert(centerIndices.end(), neighborIndices.begin(), neighborIndices.end());
  }

  std::unordered_set<int> s;
  for (int i : centerIndices)
    s.insert(i);
  centerIndices.assign( s.begin(), s.end() );
  assert(centerIndices.size() == s.size());

  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  inliers->indices = centerIndices;

  /*pcl::PointCloud<PointTypePCL>::Ptr cloudOutliers(new pcl::PointCloud<PointTypePCL>);
  pcl::ExtractIndices<PointTypePCL> extractDebug;
  extractDebug.setInputCloud(cloud);
  extractDebug.setIndices(inliers);
  extractDebug.setNegative(true);
  extractDebug.filter(*cloudOutliers);

  showCloud2(cloudOutliers, "outliers of cloud");*/

  pcl::ExtractIndices<PointTypePCL> extract;
  extract.setInputCloud(cloud);
  extract.setIndices(inliers);
  extract.setNegative(false);
  extract.filter(*cloud);

  //cout << "extracting center of cloud finished" << endl;
  //showCloud2(cloud, "center of cloud");

}

void downsampleCloud(pcl::PointCloud<PointTypePCL>::Ptr cloud, float voxelSize){
  pcl::VoxelGrid<PointTypePCL> sor;
  sor.setInputCloud (cloud);
  sor.setMinimumPointsNumberPerVoxel(1);
  sor.setLeafSize (voxelSize, voxelSize, voxelSize);
  sor.filter (*cloud);
}

void transformCloudsToSameSize(pcl::PointCloud<PointTypePCL>::Ptr cloudA, pcl::PointCloud<PointTypePCL>::Ptr cloudB){
  PointTypePCL aMin, aMax;
  pcl::getMinMax3D(*cloudA, aMin, aMax);
  PointTypePCL bMin, bMax;
  pcl::getMinMax3D(*cloudB, bMin, bMax);

  double xScale = (aMax.x - aMin.x) / (bMax.x - bMin.x);
  for (int i = 0; i < cloudA->points.size(); i++)
  {
    cloudA->points[i].x = cloudA->points[i].x / xScale;
    cloudA->points[i].y = cloudA->points[i].y / xScale;
    cloudA->points[i].z = cloudA->points[i].z / xScale;
  }
}

void substractCloudFromOtherCloud(pcl::PointCloud<PointTypePCL>::Ptr cloud, pcl::PointCloud<PointTypePCL>::Ptr otherCloud){

  pcl::KdTreeFLANN<PointTypePCL> kdtree;
  kdtree.setInputCloud (otherCloud);

  std::vector<int> indicesToBeRemoved; 

  for(int i=0; i<cloud->points.size(); ++i){
    std::vector<int> neighborIndices; //to store index of surrounding points 
    //pcl::PointIndices::Ptr neighborIndices (new pcl::PointIndices);
    std::vector<float> pointRadiusSquaredDistance; // to store distance to surrounding
    kdtree.radiusSearch(cloud->points[i], 0.25, neighborIndices, pointRadiusSquaredDistance);

    //indicesToBeRemoved[i] = neighborIndices;
    indicesToBeRemoved.insert(indicesToBeRemoved.end(), neighborIndices.begin(), neighborIndices.end());
  }

  std::vector<int>::iterator itr = indicesToBeRemoved.begin();
  std::unordered_set<int> s;
 
  for (auto curr = indicesToBeRemoved.begin(); curr != indicesToBeRemoved.end(); ++curr)
  {
      if (s.insert(*curr).second) {
          *itr++ = *curr;
      }
  }
  indicesToBeRemoved.erase(itr, indicesToBeRemoved.end());

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  inliers->indices = indicesToBeRemoved;
  removePointsInCloud(otherCloud, inliers);

}