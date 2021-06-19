#pragma once

#include "definitions.h"

#include <pcl/io/ply_io.h>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/IO/read_ply_points.h>
#include <CGAL/IO/write_ply_points.h>
#include <CGAL/property_map.h>
#include <CGAL/Aff_transformation_3.h>
#include <CGAL/aff_transformation_tags.h>

#include <CGAL/OpenGR/compute_registration_transformation.h>
#include <CGAL/OpenGR/register_point_sets.h>

#include <CGAL/pointmatcher/compute_registration_transformation.h>
#include <CGAL/pointmatcher/register_point_sets.h>

#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_3 Point_3;
typedef K::Vector_3 Vector_3;
typedef std::pair<Point_3, Vector_3> Pwn;
typedef CGAL::First_of_pair_property_map<Pwn> Point_map;
typedef CGAL::Second_of_pair_property_map<Pwn> Normal_map;

namespace params = CGAL::parameters;

void copyPointsWithNormalsPCLtoCGAL(pcl::PointCloud<PointTypePCL>::Ptr cloud, std::vector<Pwn> *pwns){
  
  size_t size = cloud->points.size();
  for(size_t i=0; i<size; ++i){
    PointTypePCL cur = cloud->points[i];
    pwns->push_back(Pwn(Point_3(cur.x, cur.y, cur.z), Vector_3(cur.normal_x, cur.normal_y, cur.normal_z)));
  }
}

void copyPointsWithNormalsCGALtoPCL(pcl::PointCloud<PointTypePCL>::Ptr cloud, std::vector<Pwn> &pwns){
  
  size_t size = pwns.size();
  for(size_t i=0; i<size; ++i){
    cloud->points[i].x = pwns[i].first.x();
    cloud->points[i].y = pwns[i].first.y();
    cloud->points[i].z = pwns[i].first.z();

    cloud->points[i].normal_x = pwns[i].second.x();
    cloud->points[i].normal_y = pwns[i].second.y();
    cloud->points[i].normal_z = pwns[i].second.z();
  }
}

int readPlyFileCGAL(std::string fileName, std::vector<Pwn> *pwns){
  std::ifstream input(fileName);
  if (!input ||
      !CGAL::read_ply_points(input, std::back_inserter(*pwns),
            params::point_map (Point_map()).
            normal_map (Normal_map())))
  {
    std::cerr << "Error: cannot read file " << fileName << std::endl;
    return -1;
  }
  input.close();

  return 0;
}

int cgalMatchingExamplePointMatcher(std::vector<Pwn> &pwns1, std::vector<Pwn> &pwns2)
{
  // Prepare ICP config
  //
  using CGAL::pointmatcher::ICP_config;
  // Possible config modules/components: https://libpointmatcher.readthedocs.io/en/latest/Configuration/#configuration-of-an-icp-chain
  // See documentation of optional named parameters for CGAL PM ICP configuration / pointmatcher config module mapping
  // Prepare point set 1 filters (PM::ReferenceDataPointsFilters)
  std::vector<ICP_config> point_set_1_filters;
  point_set_1_filters.push_back( ICP_config { /*.name=*/"MinDistDataPointsFilter"       , /*.params=*/{ {"minDist", "0.5" }}  } );
  point_set_1_filters.push_back( ICP_config { /*.name=*/"RandomSamplingDataPointsFilter", /*.params=*/{ {"prob"   , "0.05"}}  } );
  // Prepare point set 2 filters (PM::ReadingDataPointsFilters)
  std::vector<ICP_config> point_set_2_filters;
  point_set_2_filters.push_back( ICP_config { /*.name=*/"MinDistDataPointsFilter"       , /*.params=*/{ {"minDist", "0.5" }}  } );
  point_set_2_filters.push_back( ICP_config { /*.name=*/"RandomSamplingDataPointsFilter", /*.params=*/{ {"prob"   , "0.05"}}  } );
        // Prepare matcher function
  ICP_config matcher { /*.name=*/"KDTreeMatcher", /*.params=*/{ {"knn", "1"}, {"epsilon", "3.16"} } };
  // Prepare outlier filters
  std::vector<ICP_config> outlier_filters;
  outlier_filters.push_back( ICP_config { /*.name=*/"TrimmedDistOutlierFilter", /*.params=*/{ {"ratio", "0.75" }}  } );
  // Prepare error minimizer
  ICP_config error_minimizer { /*.name=*/"PointToPointErrorMinimizer"};
  // Prepare transformation checker
  std::vector<ICP_config> transformation_checkers;
  transformation_checkers.push_back( ICP_config { /*.name=*/"CounterTransformationChecker", /*.params=*/{ {"maxIterationCount", "150" }}  } );
  transformation_checkers.push_back( ICP_config { /*.name=*/"DifferentialTransformationChecker", /*.params=*/{ {"minDiffRotErr"  , "0.001" },
                                                                                                       {"minDiffTransErr", "0.01"  },
                                                                                                       {"smoothLength"   , "4"     } }
                                                } );
  // Prepare inspector
  ICP_config inspector { /*.name=*/"NullInspector" };
  // Prepare logger
  ICP_config logger { /*.name=*/"FileLogger" };
  const K::Aff_transformation_3 identity_transform = K::Aff_transformation_3(CGAL::Identity_transformation());
  // EITHER call the ICP registration method pointmatcher to get the transformation to apply to pwns2
  std::pair<K::Aff_transformation_3, bool> res =
  CGAL::pointmatcher::compute_registration_transformation
    (pwns1, pwns2,
     params::point_map(Point_map()).normal_map(Normal_map())
     .point_set_filters(point_set_1_filters)
     .matcher(matcher)
     .outlier_filters(outlier_filters)
     .error_minimizer(error_minimizer)
     .transformation_checkers(transformation_checkers)
     .inspector(inspector)
     .logger(logger),
     params::point_map(Point_map()).normal_map(Normal_map())
     .point_set_filters(point_set_2_filters)
     .transformation(identity_transform) /* initial transform for pwns2.
                                          * default value is already identity transform.
                                          * a proper initial transform could be given, for example,
                                          * a transform returned from a coarse registration algorithm.
                                          * */
     );
  bool converged = false;
  do
  {
    // OR call the ICP registration method from pointmatcher and apply the transformation to pwn2
    converged =
      CGAL::pointmatcher::register_point_sets
      (pwns1, pwns2,
       params::point_map(Point_map()).normal_map(Normal_map())
       .point_set_filters(point_set_1_filters)
       .matcher(matcher)
       .outlier_filters(outlier_filters)
       .error_minimizer(error_minimizer)
       .transformation_checkers(transformation_checkers)
       .inspector(inspector)
       .logger(logger),
       params::point_map(Point_map()).normal_map(Normal_map())
       .point_set_filters(point_set_2_filters)
       .transformation(res.first) /* pass the above computed transformation as initial transformation.
                                   * as a result, the registration will require less iterations to converge.
                                   * */
        );
    // Algorithm may randomly not converge, repeat until it does
    if (converged)
      std::cerr << "Success" << std::endl;
    else
      std::cerr << "Did not converge, try again" << std::endl;
  }
  while (!converged);
  return EXIT_SUCCESS;
}

int cgalMatchingExampleOpenGR(std::string fname1, std::string fname2)
{
  //const char* fname1 = file1Path;
  //const char* fname2 = file2Path;

  std::vector<Pwn> pwns1, pwns2;

  readPlyFileCGAL(fname1, &pwns1);
  readPlyFileCGAL(fname2, &pwns2);

  for(int i=0; i<pwns1.size(); ++i){
    std::cout <<"Point: "<< pwns1[i].first << " Vector: " << pwns1[i].second <<std::endl;
  }

  for(int i=0; i<pwns2.size(); ++i){
    std::cout <<"Point: "<< pwns2[i].first << " Vector: " << pwns2[i].second <<std::endl;
  }
  
  std::cout << "bevore calculating registration transofmration ply.\nSize of pwns1: "<<pwns1.size()<<"\nSize of pwns2: "<<pwns2.size() <<std::endl;
  // EITHER call the registration method Super4PCS from OpenGR to get the transformation to apply to pwns2
   std::pair<K::Aff_transformation_3, double> res =
    CGAL::OpenGR::compute_registration_transformation(pwns1, pwns2,
                                                      params::point_map(Point_map())
                                                      .normal_map(Normal_map())
                                                      .number_of_samples(200)
                                                      .maximum_running_time(60)
                                                      .accuracy(0.01),
                                                      params::point_map(Point_map())
                                                      .normal_map(Normal_map()));
  std::cout << "bevore register ply" <<std::endl;
  // OR call the registration method Super4PCS from OpenGR and apply the transformation to pwn2
  double score =
    CGAL::OpenGR::register_point_sets(pwns1, pwns2,
                                      params::point_map(Point_map())
                                      .normal_map(Normal_map())
                                      .number_of_samples(200)
                                      .maximum_running_time(60)
                                      .accuracy(0.01),
                                      params::point_map(Point_map())
                                      .normal_map(Normal_map()));
                                      
  std::cout << "bevore saving ply" <<std::endl;
  std::ofstream out("./tmp/pwns2_aligned.ply");
  if (!out ||
      !CGAL::write_ply_points(
        out, pwns2,
        CGAL::parameters::point_map(Point_map()).
        normal_map(Normal_map())))
  {
    return -1;
  }

  std::cout << "Registration score: " << score << ".\n"
            << "Transformed version of " << fname2
            << " written to pwn2_aligned.ply.\n";

  return 0;
}

void registerPointCloudsCGAL(pcl::PointCloud<PointTypePCL>::Ptr cloudA, pcl::PointCloud<PointTypePCL>::Ptr cloudB){

    /*std::string pathToTMP = "./tmp/";

    pcl::io::savePLYFileBinary(pathToTMP+"cloudA.ply", *cloudA);
    pcl::io::savePLYFileBinary(pathToTMP+"cloudB.ply", *cloudB);

    cgalMatchingExampleOpenGR(pathToTMP+"cloudA.ply", pathToTMP+"cloudB.ply");*/

    std::vector<Pwn> pwns1, pwns2;
    copyPointsWithNormalsPCLtoCGAL(cloudA, &pwns1);
    copyPointsWithNormalsPCLtoCGAL(cloudB, &pwns2);

    cgalMatchingExamplePointMatcher(pwns1, pwns2);
    copyPointsWithNormalsCGALtoPCL(cloudB, pwns2);
    
}
