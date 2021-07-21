#pragma once
#include <map>


//////////////////
// Registration //
//////////////////

bool registration_recalculateNormals=false;
bool registration_takeCurvatureFromNormal=false;
bool registration_debug=true;

//////////////
// IO Utils //
//////////////

bool io_utils_debug=false;

bool getBooleanFromConfig(std::map<std::string, std::string> config, std::string key){
    if(config.find(key) != config.end()){
        return config.at(key) == "true";
    } else
        std::cout << "key not found: "<<key<<std::endl;
    return false;
}