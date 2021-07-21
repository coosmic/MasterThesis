#pragma once
#include "definitions.h"
#include "utilities.h"
#include "configuration.h"

#include <stdio.h>
#include <iostream>
#include <map>
#include <string>
#include <fstream>

bool loadAsciCloud(std::string filename, pcl::PointCloud<PointTypePCL>::Ptr cloud)
{
    std::cout << "Begin Loading Model" << std::endl;
    FILE* f = fopen(filename.c_str(), "r");

    if (NULL == f)
    {
        std::cout << "ERROR: failed to open file: " << filename << std::endl;
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

bool writeShapenetFormat2(Cloud::Ptr cloud, std::string outPath, std::string name){
    ofstream filePoints, fileLabel;
    filePoints.open(outPath+"points/"+name+".pts");
    fileLabel.open(outPath+"points_label/"+name+".seg");
    if(!filePoints.is_open() || !fileLabel.is_open()){
        std::cout << "could not open file "<<outPath+"points/"+name <<std::endl;
        return false;
    }

    int colorCode;
    for(int i=0; i<cloud->size(); ++i){
        colorCode = colorToCode(cloud->points[i]);
        if(colorCode != 3){
            filePoints << cloud->points[i].x << " " 
            << cloud->points[i].y << " " 
            << cloud->points[i].z << "\n";
            fileLabel << colorCode << "\n";
        }
    }

    filePoints.close();
    fileLabel.close();
    return true;
}

bool writeShapenetFormat(Cloud::Ptr cloud, std::string outPath){
    ofstream file;
    file.open(outPath+".txt");
    if(!file.is_open()){
        return false;
    }

    int colorCode;
    for(int i=0; i<cloud->size(); ++i){
        colorCode = colorToCode(cloud->points[i]);
        if(colorCode != BackgroundLabel)
            file << cloud->points[i].x << " " 
            << cloud->points[i].y << " " 
            << cloud->points[i].z << " " 
            <<  cloud->points[i].normal_x << " " 
            <<  cloud->points[i].normal_y << " " 
            <<  cloud->points[i].normal_z << " " 
            << colorCode << "\n";
    }

    file.close();

    std::cout << "Exported " << outPath+".txt" << " in Shapenet Format\n";

    return true;
}

std::map<std::string, std::string> readConfig(std::string path){
    std::map<std::string, std::string> config;
    std::ifstream infile(path);
    if (!infile.is_open())
    {
        std::cout << "ERROR: failed to open config: " << path << std::endl;
        return config;
    }

    std::string key, value, line;
    while (std::getline(infile, line)){
        int index = line.find("=");
        if (index == -1){
            if(io_utils_debug)
                std::cout << "malformed line "<<line<<" skipped\n";
            continue;
        }
        
        //std::cout << index << ":" <<line.length()<<std::endl;
        key = line.substr(0, index);
        value = line.substr(index+1, line.length()-1);
        if(io_utils_debug)
            std::cout << "adding "<<key<<"="<<value<<" to config\n";
        config.insert(std::make_pair(key, value));
    }
    return config;
}