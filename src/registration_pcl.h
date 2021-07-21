#pragma once
#include <iostream>   

#include "definitions.h"
#include "utilities.h"

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/principal_curvatures.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/transformation_estimation_svd_scale.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>

#include <math.h>  

#include "debug.h"

// --------------------
// -----Parameters-----
// --------------------
// SIFT Keypoint parameters
const float min_scale = 0.02f; // the standard deviation of the smallest scale in the scale space
const int n_octaves = 3;  // the number of octaves (i.e. doublings of scale) to compute
const int n_scales_per_octave = 4; // the number of scales to compute within each octave
const float min_contrast = 0.001f; // the minimum contrast required for detection

// Sample Consensus Initial Alignment parameters (explanation below)
const float min_sample_dist = 0.0125f;  //org 0.025
const float max_correspondence_dist = 0.01f;
const int nr_iters = 500;

// ICP parameters (explanation below)
const float max_correspondence_distance = 0.05f;
const float outlier_rejection_threshold = 0.05f;
const float transformation_epsilon = 0;
const int max_iterations = 100;
  

void
setViewerPose (pcl::visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
{
  Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f (0, 0, 0);
  Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector;
  Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f (0, -1, 0);
  viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
                            look_at_vector[0], look_at_vector[1], look_at_vector[2],   
                            up_vector[0], up_vector[1], up_vector[2]); 
}

/* Use SampleConsensusInitialAlignment to find a rough alignment from the source cloud to the target cloud by fin    ding
 * correspondences between two sets of local features
 * Inputs:
 *   source_points
 *     The "source" points, i.e., the points that must be transformed to align with the target point cloud
 *   source_descriptors
 *     The local descriptors for each source point
 *   target_points
 *     The "target" points, i.e., the points to which the source point cloud will be aligned                     
 *   target_descriptors
 *     The local descriptors for each target point
 *   min_sample_distance
 *     The minimum distance between any two random samples
 *   max_correspondence_distance
 *     The maximum distance between a point and its nearest neighbor correspondent in order to be considered
 *     in the alignment process
 *   nr_interations
 *     The number of RANSAC iterations to perform
 * Return: A transformation matrix that will roughly align the points in source to the points in target
 */
typedef pcl::PointWithScale PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;
typedef pcl::FPFHSignature33 LocalDescriptorT;
typedef pcl::PointCloud<LocalDescriptorT>::Ptr LocalDescriptorsPtr;
Eigen::Matrix4f
computeInitialAlignment (const PointCloudPtr & source_points, const LocalDescriptorsPtr & source_descriptors,
                         const PointCloudPtr & target_points, const LocalDescriptorsPtr & target_descriptors,
                         float min_sample_distance, float max_correspondence_distance, int nr_iterations)
{
  pcl::SampleConsensusInitialAlignment<PointT, PointT, LocalDescriptorT> sac_ia;
  sac_ia.setMinSampleDistance (min_sample_distance);
  sac_ia.setMaxCorrespondenceDistance (max_correspondence_distance);
  sac_ia.setMaximumIterations (nr_iterations);

  sac_ia.setInputCloud (source_points);
  sac_ia.setSourceFeatures (source_descriptors);

  sac_ia.setInputTarget (target_points);
  sac_ia.setTargetFeatures (target_descriptors);

  PointCloud registration_output;
  sac_ia.align (registration_output);

  return (sac_ia.getFinalTransformation ());
}

/* Use IterativeClosestPoint to find a precise alignment from the source cloud to the target cloud,                   
 * starting with an intial guess
 * Inputs:
 *   source_points
 *     The "source" points, i.e., the points that must be transformed to align with the target point cloud
 *   target_points
 *     The "target" points, i.e., the points to which the source point cloud will be aligned
 *   intial_alignment
 *     An initial estimate of the transformation matrix that aligns the source points to the target points
 *   max_correspondence_distance
 *     A threshold on the distance between any two corresponding points.  Any corresponding points that are further 
 *     apart than this threshold will be ignored when computing the source-to-target transformation
 *   outlier_rejection_threshold
 *     A threshold used to define outliers during RANSAC outlier rejection
 *   transformation_epsilon
 *     The smallest iterative transformation allowed before the algorithm is considered to have converged
 *   max_iterations
 *     The maximum number of ICP iterations to perform
 * Return: A transformation matrix that will precisely align the points in source to the points in target
 */
typedef PointTypeRegistration ICPPointT;
typedef pcl::PointCloud<ICPPointT> ICPPointCloud;
typedef pcl::PointCloud<ICPPointT>::Ptr ICPPointCloudPtr;
Eigen::Matrix4f
refineAlignment (const ICPPointCloudPtr & source_points, const ICPPointCloudPtr & target_points,
                 const Eigen::Matrix4f initial_alignment, float max_correspondence_distance,
                 float outlier_rejection_threshold, float transformation_epsilon, float max_iterations) {

  pcl::IterativeClosestPoint<ICPPointT, ICPPointT> icp;
  icp.setMaxCorrespondenceDistance (max_correspondence_distance);
  icp.setRANSACOutlierRejectionThreshold (outlier_rejection_threshold);
  icp.setTransformationEpsilon (transformation_epsilon);
  icp.setMaximumIterations (max_iterations);

  ICPPointCloudPtr source_points_transformed (new ICPPointCloud);
  pcl::transformPointCloud (*source_points, *source_points_transformed, initial_alignment);

  icp.setInputCloud (source_points_transformed);
  icp.setInputTarget (target_points);

  ICPPointCloud registration_output;
  icp.align (registration_output);

  return (icp.getFinalTransformation () * initial_alignment);
}

Eigen::Matrix4f registerClouds(pcl::PointCloud<PointTypeRegistration>::Ptr source_cloud_ptr, pcl::PointCloud<PointTypeRegistration>::Ptr target_cloud_ptr, bool transformSourceCloud){

    pcl::PointCloud<PointTypeRegistration>& source_cloud = *source_cloud_ptr;
    pcl::PointCloud<PointTypeRegistration>& target_cloud = *target_cloud_ptr;

    Eigen::Affine3f scene_sensor_pose (Eigen::Affine3f::Identity ());
    scene_sensor_pose = Eigen::Affine3f (Eigen::Translation3f (source_cloud.sensor_origin_[0],
                                                               source_cloud.sensor_origin_[1],
                                                               source_cloud.sensor_origin_[2])) *
                        Eigen::Affine3f (source_cloud.sensor_orientation_);
    

    // Estimate cloud normals
    cout << "Computing source cloud normals\n";
    pcl::NormalEstimation<PointTypeRegistration, pcl::PointNormal> ne;
    pcl::PointCloud<pcl::PointNormal>::Ptr src_normals_ptr (new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointNormal>& src_normals = *src_normals_ptr;
    pcl::search::KdTree<PointTypeRegistration>::Ptr tree_xyz (new pcl::search::KdTree<PointTypeRegistration>());
    ne.setInputCloud(source_cloud_ptr);
    ne.setSearchMethod(tree_xyz);
    ne.setRadiusSearch(0.05);
    ne.compute(*src_normals_ptr);
    for(size_t i = 0;  i < src_normals.points.size(); ++i) {
        src_normals.points[i].x = source_cloud.points[i].x;
        src_normals.points[i].y = source_cloud.points[i].y;
        src_normals.points[i].z = source_cloud.points[i].z;
    }
    //showCloud2(source_cloud_ptr, "src normals", src_normals_ptr_2);

    cout << "Computing target cloud normals\n";
    pcl::PointCloud<pcl::PointNormal>::Ptr tar_normals_ptr (new pcl::PointCloud<pcl::PointNormal>);
    pcl::PointCloud<pcl::PointNormal>& tar_normals = *tar_normals_ptr;
    ne.setInputCloud(target_cloud_ptr);
    ne.compute(*tar_normals_ptr);
    for(size_t i = 0;  i < tar_normals.points.size(); ++i) {
        tar_normals.points[i].x = target_cloud.points[i].x;
        tar_normals.points[i].y = target_cloud.points[i].y;
        tar_normals.points[i].z = target_cloud.points[i].z;
    }
    //showCloud2(target_cloud_ptr, "src normals", tar_normals_ptr);

    int nanCount=0;
    int differingPoints =0;
    for(int i=0; i<tar_normals.size(); ++i){
      if(isnan(tar_normals.points[i].normal_x))
        ++nanCount;
      if(tar_normals.points[i].x != tar_normals_ptr->points[i].x || tar_normals.points[i].y != tar_normals_ptr->points[i].y || tar_normals.points[i].z != tar_normals_ptr->points[i].z  )
        ++differingPoints;
    }
    cout << "Found "<<nanCount<<" nan normals of "<<tar_normals.size()<<"\n";
    cout << "Found "<<differingPoints<<" differing points of "<<tar_normals.size()<<"\n";

    // Estimate the SIFT keypoints
    pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
    pcl::PointCloud<pcl::PointWithScale>::Ptr src_keypoints_ptr (new pcl::PointCloud<pcl::PointWithScale>);
    pcl::PointCloud<pcl::PointWithScale>& src_keypoints = *src_keypoints_ptr;
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree_normal(new pcl::search::KdTree<pcl::PointNormal> ());
    sift.setSearchMethod(tree_normal);
    sift.setScales(min_scale, n_octaves, n_scales_per_octave);
    sift.setMinimumContrast(min_contrast);
    sift.setInputCloud(src_normals_ptr);
    sift.compute(src_keypoints);
    cout << "Found " << src_keypoints.points.size () << " SIFT keypoints in source cloud\n";
    pcl::PointCloud<pcl::PointWithScale>::Ptr tar_keypoints_ptr (new pcl::PointCloud<pcl::PointWithScale>);
    pcl::PointCloud<pcl::PointWithScale>& tar_keypoints = *tar_keypoints_ptr;
    sift.setInputCloud(tar_normals_ptr);
    sift.compute(tar_keypoints);
    cout << "Found " << tar_keypoints.points.size () << " SIFT keypoints in target cloud\n";

    // Extract FPFH features from SIFT keypoints
    //pcl::search::KdTree<PointTypeRegistration>::Ptr tree_xyz (new pcl::search::KdTree<PointTypeRegistration>());
    pcl::PointCloud<PointTypeRegistration>::Ptr src_keypoints_xyz (new pcl::PointCloud<PointTypeRegistration>);                           
    pcl::copyPointCloud (src_keypoints, *src_keypoints_xyz);
    pcl::FPFHEstimation<PointTypeRegistration, pcl::PointNormal, pcl::FPFHSignature33> fpfh;
    fpfh.setSearchSurface (source_cloud_ptr);
    fpfh.setInputCloud (src_keypoints_xyz);
    fpfh.setInputNormals (src_normals_ptr);
    fpfh.setSearchMethod (tree_xyz);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr src_features_ptr (new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::PointCloud<pcl::FPFHSignature33>& src_features = *src_features_ptr;
    fpfh.setRadiusSearch(0.05);
    fpfh.compute(src_features);
    cout << "Computed " << src_features.size() << " FPFH features for source cloud\n";
    pcl::PointCloud<PointTypeRegistration>::Ptr tar_keypoints_xyz (new pcl::PointCloud<PointTypeRegistration>);                           
    pcl::copyPointCloud (tar_keypoints, *tar_keypoints_xyz);
    fpfh.setSearchSurface (target_cloud_ptr);
    fpfh.setInputCloud (tar_keypoints_xyz);
    fpfh.setInputNormals (tar_normals_ptr);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr tar_features_ptr (new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::PointCloud<pcl::FPFHSignature33>& tar_features = *tar_features_ptr;
    fpfh.compute(tar_features);
    cout << "Computed " << tar_features.size() << " FPFH features for target cloud\n";

    // Compute the transformation matrix for alignment
    Eigen::Matrix4f tform = Eigen::Matrix4f::Identity();
    tform = computeInitialAlignment (src_keypoints_ptr, src_features_ptr, tar_keypoints_ptr,
            tar_features_ptr, min_sample_dist, max_correspondence_dist, nr_iters);

    
    /*tform = refineAlignment (source_cloud_ptr, target_cloud_ptr, tform, max_correspondence_distance,
            outlier_rejection_threshold, transformation_epsilon, max_iterations);*/
    
    if(transformSourceCloud){
      pcl::transformPointCloud(source_cloud, source_cloud, tform);
      cout << "Calculated transformation\n";
    }
    

    return tform;
    
}

void calculatePrinzipalCurvature(pcl::PointCloud<PointTypePCL>::Ptr cloud_ptr){
  pcl::PrincipalCurvaturesEstimation<PointTypePCL, PointTypePCL, pcl::PrincipalCurvatures> principalCurvaturesEstimation;
  principalCurvaturesEstimation.setInputCloud (cloud_ptr);
  principalCurvaturesEstimation.setInputNormals (cloud_ptr);

  pcl::search::KdTree<PointTypePCL>::Ptr tree (new pcl::search::KdTree<PointTypePCL>);
  principalCurvaturesEstimation.setSearchMethod (tree);
  principalCurvaturesEstimation.setRadiusSearch(1.0);

  // Actually compute the principal curvatures
  pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr principalCurvatures (new pcl::PointCloud<pcl::PrincipalCurvatures> ());
  principalCurvaturesEstimation.compute (*principalCurvatures);

  #pragma omp parallel for
  for(int i=0; i< cloud_ptr->size(); ++i){
    //cloud_ptr->points[i].curvature = (principalCurvatures->points[i].principal_curvature_x + principalCurvatures->points[i].principal_curvature_y + principalCurvatures->points[i].principal_curvature_z)*0.33333333;
    cloud_ptr->points[i].curvature = principalCurvatures->points[i].pc2 / (principalCurvatures->points[i].pc2 + principalCurvatures->points[i].pc1);
  }

}

void getCurvatureFromNormalEstimation(Cloud::Ptr cloud){
  pcl::NormalEstimation<PointTypePCL, pcl::PointNormal> ne;
  pcl::PointCloud<pcl::PointNormal>::Ptr normals_ptr (new pcl::PointCloud<pcl::PointNormal>);
  pcl::search::KdTree<PointTypePCL>::Ptr tree_xyz (new pcl::search::KdTree<PointTypePCL>());
  ne.setInputCloud(cloud);
  ne.setSearchMethod(tree_xyz);
  ne.setRadiusSearch(0.01);
  ne.compute(*normals_ptr);
  #pragma omp parallel for
  for(size_t i = 0;  i < cloud->points.size(); ++i) {
      cloud->points[i].curvature = normals_ptr->points[i].curvature;
  }
}

Eigen::Matrix4f registerClouds(
  pcl::PointCloud<PointTypePCL>::Ptr source_cloud_ptr, 
  pcl::PointCloud<PointTypePCL>::Ptr target_cloud_ptr, 
  bool transformSourceCloud){

    cout << "Estimating Curvature for src cloud\n";
    registration_takeCurvatureFromNormal? getCurvatureFromNormalEstimation(source_cloud_ptr):calculatePrinzipalCurvature(source_cloud_ptr);
    cout << "Estimating Curvature for target cloud\n";
    registration_takeCurvatureFromNormal? getCurvatureFromNormalEstimation(target_cloud_ptr):calculatePrinzipalCurvature(target_cloud_ptr);

    // Estimate the SIFT keypoints
    pcl::SIFTKeypoint<PointTypePCL, pcl::PointWithScale> sift;
    pcl::PointCloud<pcl::PointWithScale>::Ptr src_keypoints_ptr (new pcl::PointCloud<pcl::PointWithScale>);
    pcl::PointCloud<pcl::PointWithScale>& src_keypoints = *src_keypoints_ptr;
    pcl::search::KdTree<PointTypePCL>::Ptr tree_normal(new pcl::search::KdTree<PointTypePCL> ());
    sift.setSearchMethod(tree_normal);
    sift.setScales(min_scale, n_octaves, n_scales_per_octave);
    sift.setMinimumContrast(min_contrast);
    sift.setInputCloud(source_cloud_ptr);
    sift.compute(src_keypoints);
    cout << "Found " << src_keypoints.points.size () << " SIFT keypoints in source cloud\n";
    pcl::PointCloud<pcl::PointWithScale>::Ptr tar_keypoints_ptr (new pcl::PointCloud<pcl::PointWithScale>);
    pcl::PointCloud<pcl::PointWithScale>& tar_keypoints = *tar_keypoints_ptr;
    sift.setInputCloud(target_cloud_ptr);
    sift.compute(tar_keypoints);
    cout << "Found " << tar_keypoints.points.size () << " SIFT keypoints in target cloud\n";

    // Extract FPFH features from SIFT keypoints
    pcl::search::KdTree<PointTypePCL>::Ptr tree_xyz (new pcl::search::KdTree<PointTypePCL>());
    pcl::PointCloud<PointTypePCL>::Ptr src_keypoints_xyz (new pcl::PointCloud<PointTypePCL>);                           
    pcl::copyPointCloud (src_keypoints, *src_keypoints_xyz);
    pcl::FPFHEstimation<PointTypePCL, PointTypePCL, pcl::FPFHSignature33> fpfh;
    fpfh.setSearchSurface (source_cloud_ptr);
    fpfh.setInputCloud (src_keypoints_xyz);
    fpfh.setInputNormals (source_cloud_ptr);
    fpfh.setSearchMethod (tree_xyz);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr src_features_ptr (new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::PointCloud<pcl::FPFHSignature33>& src_features = *src_features_ptr;
    fpfh.setRadiusSearch(0.05);
    fpfh.compute(src_features);
    cout << "Computed " << src_features.size() << " FPFH features for source cloud\n";
    pcl::PointCloud<PointTypePCL>::Ptr tar_keypoints_xyz (new pcl::PointCloud<PointTypePCL>);                           
    pcl::copyPointCloud (tar_keypoints, *tar_keypoints_xyz);
    fpfh.setSearchSurface (target_cloud_ptr);
    fpfh.setInputCloud (tar_keypoints_xyz);
    fpfh.setInputNormals (target_cloud_ptr);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr tar_features_ptr (new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::PointCloud<pcl::FPFHSignature33>& tar_features = *tar_features_ptr;
    fpfh.compute(tar_features);
    cout << "Computed " << tar_features.size() << " FPFH features for target cloud\n";

    // Compute the transformation matrix for alignment
    Eigen::Matrix4f tform = Eigen::Matrix4f::Identity();
    tform = computeInitialAlignment (src_keypoints_ptr, src_features_ptr, tar_keypoints_ptr,
            tar_features_ptr, min_sample_dist, max_correspondence_dist, nr_iters);

    
    /*tform = refineAlignment (source_cloud_ptr, target_cloud_ptr, tform, max_correspondence_distance,
            outlier_rejection_threshold, transformation_epsilon, max_iterations);*/
    
    if(transformSourceCloud){
      pcl::transformPointCloud(*source_cloud_ptr, *source_cloud_ptr, tform);
      cout << "Calculated transformation\n";
    }
    

    return tform;
}

void registrationPipeline(Cloud::Ptr cloudSrc, Cloud::Ptr cloudTrg, bool useSift = true, bool useScale = true, bool useCgalICP = true, bool usePclICP = true, bool removePlanes = true){
  cout << "register clouds feature based..." <<endl;

  if(registration_recalculateNormals){
    
    pcl::PointCloud<PointTypeRegistration>::Ptr src(new pcl::PointCloud<PointTypeRegistration>);
    pcl::PointCloud<pcl::PointNormal>::Ptr src_normals(new pcl::PointCloud<pcl::PointNormal>);
    copyPointCloud(*cloudSrc, *src);
    //copyPointCloud(*cloudA, *src_normals);
    pcl::PointCloud<PointTypeRegistration>::Ptr target(new pcl::PointCloud<PointTypeRegistration>);
    pcl::PointCloud<pcl::PointNormal>::Ptr target_normals(new pcl::PointCloud<pcl::PointNormal>);
    copyPointCloud(*cloudTrg, *target);
    Eigen::Matrix4f initialTransformation;
    if(useSift){
      //copyPointCloud(*cloudB, *target_normals);
      target_normals->resize(cloudTrg->size());
      src_normals->resize(cloudSrc->size());

      //showCloud2(target, "Target Normals", target_normals);

      initialTransformation = registerClouds(src, target, true);

      //debug_showCombinedCloud(src, target, "Feature Registration");
    }
    
    Eigen::Matrix4f icpTransformation;
    if(useScale){
      cout << "register clouds with icp and estimate scale..." <<endl;
      pcl::IterativeClosestPoint<PointTypeRegistration, PointTypeRegistration> icp2;
      icp2.setInputSource(src);
      icp2.setInputTarget(target);

      boost::shared_ptr<pcl::registration::TransformationEstimationSVDScale <PointTypeRegistration, PointTypeRegistration>> teSVDscale (new pcl::registration::TransformationEstimationSVDScale <PointTypeRegistration, PointTypeRegistration>());
      icp2.setTransformationEstimation (teSVDscale);

      pcl::PointCloud<PointTypeRegistration> Unused2;
      icp2.align(Unused2);

      icpTransformation = icp2.getFinalTransformation();
      pcl::transformPointCloud (*src, *src, icpTransformation); 
    }
    
    Eigen::Matrix4f finalTransformation = icpTransformation * initialTransformation;
    pcl::transformPointCloud (*cloudSrc, *cloudSrc, finalTransformation);

  } else {
    if(useSift)
      Eigen::Matrix4f initialTransformation = registerClouds(cloudSrc, cloudTrg, true);
    if(registration_debug)debug_showCombinedCloud(cloudSrc, cloudTrg, "Feature Registration");

    /*cout << "remove planes and rescale..." <<endl;
    planeFilter(cloudA);
    planeFilter(cloudB);
    transformCloudsToSameSize(cloudA, cloudB);
    debug_showCombinedCloud(cloudA, cloudB, "rescaled Clouds without Plane");*/
    if(useScale){
      cout << "register clouds with icp and estimate scale..." <<endl;
      pcl::IterativeClosestPoint<PointTypePCL, PointTypePCL> icp2;
      icp2.setInputSource(cloudSrc);
      icp2.setInputTarget(cloudTrg);
      //icp2.setMaxCorrespondenceDistance(1.0);

      boost::shared_ptr<pcl::registration::TransformationEstimationSVDScale <PointTypePCL, PointTypePCL>> teSVDscale (new pcl::registration::TransformationEstimationSVDScale <PointTypePCL, PointTypePCL>());
      icp2.setTransformationEstimation (teSVDscale);

      pcl::PointCloud<PointTypePCL> Unused2;
      icp2.align(Unused2);

      Eigen::Matrix4f icpTransformation = icp2.getFinalTransformation();
      pcl::transformPointCloud (*cloudSrc, *cloudSrc, icpTransformation);
    }
  }
  if(registration_debug)debug_showCombinedCloud(cloudSrc, cloudTrg, "scale alligned Clouds");

  //Eigen::Matrix4f finalTransformation = icpTransformation * initialTransformation;
  //pcl::transformPointCloud (*cloudA, *cloudA, finalTransformation);
  if(useCgalICP)
    registerPointCloudsCGAL(cloudSrc, cloudTrg);

  if(removePlanes){
    //remove planes
    planeFilter(cloudSrc);
    planeFilter(cloudTrg);
    if(registration_debug)debug_showCombinedCloud(cloudSrc, cloudTrg, "plane Filtered Cloud");
  }

  cout << "register clouds with icp..." <<endl;
  if(usePclICP){
    pcl::IterativeClosestPoint<PointTypePCL, PointTypePCL> icp;
    icp.setInputSource(cloudSrc);
    icp.setInputTarget(cloudTrg);

    pcl::PointCloud<PointTypePCL> Unused;
    icp.align(Unused);

    pcl::transformPointCloud (*cloudSrc, *cloudSrc, icp.getFinalTransformation());
  }
  
}