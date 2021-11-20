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
#include <pcl/filters/random_sample.h>

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
    
    //cout << "centroid: " << centroid[0] << " " <<  centroid[1] << " " <<   centroid[2] << " " <<   centroid[3] << " \n";

  transform_2.translation() << -centroid[0], -centroid[1], -centroid[2];
  transform_2.rotate (Eigen::AngleAxisf (theta, rotation_vector));
  //std::cout << "Transformation matrix: " << std::endl << transform_2.matrix() << std::endl;

  pcl::transformPointCloud (*cloud, *cloud, transform_2);

  //showCloud2(cloud, "rotated PointCloud v2", coefficientsPlane);

}

void innerNoiseFilter(pcl::PointCloud<PointTypePCL>::Ptr cloud, int minNumberNeighbors, float radius){
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

void noiseFilter(pcl::PointCloud<PointTypePCL>::Ptr cloud, int minNeighbors1=50, float radius1=0.08, int minNeighbors2=1400, float radius2=0.2){

  //////////////////////////
  // Remove Noise Level 1 //
  //////////////////////////
  if(minNeighbors1 > 0)
    innerNoiseFilter(cloud, minNeighbors1, radius1);

  //////////////////////////
  // Remove Noise Level 2 //
  //////////////////////////
  if(minNeighbors2 > 0)
    innerNoiseFilter(cloud, minNeighbors2, radius2);
}

void findPlaneInCloud (pcl::PointCloud<PointTypePCL>::Ptr cloud, pcl::ModelCoefficients::Ptr coefficients, pcl::PointIndices::Ptr inliers){
  // Create the segmentation object
  pcl::SACSegmentation<PointTypePCL> seg;
  // Optional
  seg.setOptimizeCoefficients (true);
  // Mandatory
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.02);

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

void transformCloudToUnitSize(Cloud::Ptr cloud, float scale){
  PointTypePCL min, max;
  pcl::getMinMax3D(*cloud, min, max);

  float length = abs(min.x) + abs(max.x);
  float width = abs(min.y) + abs(max.y);

  float faktor = scale;
  if(length > width)
    faktor = length;
  else
    faktor = width;

  for (int i = 0; i < cloud->points.size(); i++)
  {
    cloud->points[i].x = cloud->points[i].x / faktor * scale;
    cloud->points[i].y = cloud->points[i].y / faktor * scale;
    cloud->points[i].z = cloud->points[i].z / faktor * scale;
  }
  
}

void centerCloud(Cloud::Ptr cloud){
  pcl::PCA<PointTypePCL> pca;
  pca.setInputCloud(cloud);
  Eigen::Vector4f meanVector = pca.getMean();
  Eigen::Affine3f transform(Eigen::Translation3f(-meanVector.x(),-meanVector.y(),-meanVector.z()));
  Eigen::Matrix4f matrix = transform.matrix();
  pcl::transformPointCloud (*cloud, *cloud, matrix);
}

Cloud::Ptr getPointsNearCloudFromOtherCloud(pcl::PointCloud<PointTypePCL>::Ptr cloud, pcl::PointCloud<PointTypePCL>::Ptr otherCloud, float radius){
  pcl::KdTreeFLANN<PointTypePCL> kdtree;
  kdtree.setInputCloud (otherCloud);

  std::vector<int> indicesToBeTaken; 

  for(int i=0; i<cloud->points.size(); ++i){
    std::vector<int> neighborIndices; //to store index of surrounding points 
    //pcl::PointIndices::Ptr neighborIndices (new pcl::PointIndices);
    std::vector<float> pointRadiusSquaredDistance; // to store distance to surrounding
    kdtree.radiusSearch(cloud->points[i], radius, neighborIndices, pointRadiusSquaredDistance);

    //indicesToBeRemoved[i] = neighborIndices;
    indicesToBeTaken.insert(indicesToBeTaken.end(), neighborIndices.begin(), neighborIndices.end());
  }

  std::vector<int>::iterator itr = indicesToBeTaken.begin();
  std::unordered_set<int> s;
 
  for (auto curr = indicesToBeTaken.begin(); curr != indicesToBeTaken.end(); ++curr)
  {
      if (s.insert(*curr).second) {
          *itr++ = *curr;
      }
  }
  indicesToBeTaken.erase(itr, indicesToBeTaken.end());

  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  inliers->indices = indicesToBeTaken;

  Cloud::Ptr out (new Cloud);
  pcl::ExtractIndices<PointTypePCL> extract;
  extract.setInputCloud(otherCloud);
  extract.setIndices(inliers);
  extract.setNegative(false);
  extract.filter(*out);
  return out;
}

void substractCloudFromOtherCloud(pcl::PointCloud<PointTypePCL>::Ptr cloud, pcl::PointCloud<PointTypePCL>::Ptr otherCloud, float radius){

  pcl::KdTreeFLANN<PointTypePCL> kdtree;
  kdtree.setInputCloud (otherCloud);

  std::vector<int> indicesToBeRemoved; 

  for(int i=0; i<cloud->points.size(); ++i){
    std::vector<int> neighborIndices; //to store index of surrounding points 
    //pcl::PointIndices::Ptr neighborIndices (new pcl::PointIndices);
    std::vector<float> pointRadiusSquaredDistance; // to store distance to surrounding
    kdtree.radiusSearch(cloud->points[i], radius, neighborIndices, pointRadiusSquaredDistance);

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

inline
uint8_t setColorChannelAt(Cloud::Ptr cloud, ColorChannel channel, int index, uint8_t value){
  uint8_t tmp;
  switch(channel){
    case r:
      tmp = cloud->points[index].r;
      cloud->points[index].r = value;
    break;
    case g:
      tmp = cloud->points[index].g;
      cloud->points[index].g = value;
    break;
    case b:
      tmp = cloud->points[index].b;
      cloud->points[index].b = value;
    break;
  }
  return tmp;
}

inline 
void swapColorChannelAt(Cloud::Ptr cloud, ColorChannel channelSrc, ColorChannel channelTgt, int index){
  switch(channelSrc){
    case r:
      setColorChannelAt(cloud, channelSrc, index, setColorChannelAt(cloud, channelTgt, index, cloud->points[index].r));
    break;
    case g:
      setColorChannelAt(cloud, channelSrc, index,setColorChannelAt(cloud, channelTgt, index, cloud->points[index].g));
    break;
    case b:
      setColorChannelAt(cloud, channelSrc, index, setColorChannelAt(cloud, channelTgt, index, cloud->points[index].b));
    break;
  }
}

void switchColorChannel(Cloud::Ptr cloud, ColorChannel channelSrc, ColorChannel channelTgt){
  #pragma omp parallel for
  for(int i=0; i< cloud->points.size(); ++i){
    swapColorChannelAt(cloud, channelSrc, channelTgt, i);
  }
}

void setColorChannelExclusive(Cloud::Ptr cloud, ColorChannel channel, uint8_t value){
  ColorChannel otherChannel1 = channel == ColorChannel::b ? ColorChannel::r : channel == ColorChannel::r ? ColorChannel::g : ColorChannel::b;
  ColorChannel otherChannel2 =  channel == ColorChannel::b && otherChannel1 == ColorChannel::g ? ColorChannel::r : 
                                channel == ColorChannel::b && otherChannel1 == ColorChannel::r ? ColorChannel::g : 
                                channel == ColorChannel::r && otherChannel1 == ColorChannel::g ? ColorChannel::b : 
                                channel == ColorChannel::r && otherChannel1 == ColorChannel::b ? ColorChannel::g : 
                                channel == ColorChannel::g && otherChannel1 == ColorChannel::b ? ColorChannel::r : 
                                ColorChannel::b;
  #pragma omp parallel for
  for(int i=0; i< cloud->points.size(); ++i){
    setColorChannelAt(cloud, channel, i, value);
    //setColorChannelAt(cloud, otherChannel1, i, 0);
    //setColorChannelAt(cloud, otherChannel2, i, 0);
  }
}

inline
int colorToCode(PointTypePCL point){
    
    // background
    if(point.r == 255 && point.g == 0 && point.b == 0)
        return BackgroundLabel;
    // stem
    if(point.r == 0 && point.g == 255 && point.b == 0)
        return StemLabel;
    // leave
    return LeaveLabel;
}

/**
 * Removes background points from cloud by color. Returns a new pointcloud containing the background
 */
Cloud::Ptr removeBackgroundPointsShapenet(Cloud::Ptr cloud){
  pcl::PointIndices::Ptr backgroundIndicies(new pcl::PointIndices());
  for(int i=0; i<cloud->size(); ++i){
    if(colorToCode(cloud->points[i]) == BackgroundLabel){
      backgroundIndicies->indices.push_back(i);
    }
  }

  Cloud::Ptr background (new Cloud);

  pcl::ExtractIndices<PointTypePCL> extract;
  extract.setInputCloud(cloud);
  extract.setIndices(backgroundIndicies);
  extract.filter(*background);

  extract.setNegative(true);
  extract.filter(*cloud);

  return background;

}

void transformToShapenetFormat(Cloud::Ptr cloud, Eigen::Matrix4f &translation, Eigen::Matrix4f &scale){
  
  PointTypePCL min, max;
  pcl::getMinMax3D(*cloud, min, max);

  Eigen::Affine3f transform(Eigen::Translation3f(-min.x,-min.y,-min.z));
  Eigen::Matrix4f matrix = transform.matrix();
  translation = matrix;
  pcl::transformPointCloud (*cloud, *cloud, matrix);
  //showCloud2(cloud, "translation");

  pcl::getMinMax3D(*cloud, min, max);

  float scaleFactor = std::max(max.x, std::max(max.y,max.z));
  scaleFactor = 1/scaleFactor;

  Eigen::Transform <float , 3, Eigen::Affine > t = Eigen::Transform <float , 3, Eigen::Affine >::Identity ();
  t.scale( scaleFactor );
  
  matrix = t.matrix();
  scale = matrix;
  pcl::transformPointCloud (*cloud, *cloud, matrix);
}

Cloud::Ptr subSampleCloudRandom(Cloud::Ptr cloud, int numberOfSamples){
  Cloud::Ptr subsampledCloud (new Cloud);
  
  pcl::RandomSample<PointTypePCL> rs;
  rs.setInputCloud(cloud);
  rs.setSample(numberOfSamples);
  rs.setSeed(rand());

  std::vector<int> indices;
  rs.filter(indices);

  //Extract Subsamples from Cloud
  pcl::PointIndices::Ptr subsampledIndicies(new pcl::PointIndices());
  subsampledIndicies->indices = indices;
  pcl::ExtractIndices<PointTypePCL> extract;
  extract.setInputCloud(cloud);
  extract.setIndices(subsampledIndicies);
  extract.filter(*subsampledCloud);

  //Remove Subsamples from Cloud
  extract.setNegative(true);
  extract.filter(*cloud);

  return subsampledCloud;
}
