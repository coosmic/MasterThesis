import os
import sys
import numpy as np
import open3d as o3d

import utilities

def robustICP(srcPath, targetPath, outPath, scale, srcCloud, targetCloud):
    
    if not os.path.isdir(outPath):
        os.makedirs(outPath)

    command = f"../build/FRICP {srcPath} {targetPath} {outPath} 3 {scale}"
    os.system(command)

    pathToTransformation = os.path.join(outPath, "m3trans.txt")

    # Apply Transformation
    transformation = np.loadtxt(pathToTransformation).astype(np.float32)
    #srcCloud = np.loadtxt(srcPath).astype(np.float32)
    #targetCloud = np.loadtxt(targetPath).astype(np.float32)
    
    srcCloud *= scale
    transformedSrc = np.hstack((srcCloud, np.ones((srcCloud.shape[0], 1))))  #(nx3)->(nx4)
    transformedSrc = transformedSrc.dot(transformation.T)[:, 0:3]

    # Calc and return error
    return utilities.pointCloudDistance(transformedSrc, targetCloud), transformation

def display_open3d(template, source, transformed_source):
    template_ = o3d.geometry.PointCloud()
    source_ = o3d.geometry.PointCloud()
    transformed_source_ = o3d.geometry.PointCloud()
    template_.points = o3d.utility.Vector3dVector(template)
    source_.points = o3d.utility.Vector3dVector(source + np.array([0,0,0]))
    transformed_source_.points = o3d.utility.Vector3dVector(transformed_source)
    template_.paint_uniform_color([1, 0, 0])
    source_.paint_uniform_color([0, 1, 0])
    transformed_source_.paint_uniform_color([0, 0, 1])
    #o3d.visualization.draw_geometries([template_, source_, transformed_source_])
    o3d.visualization.draw_geometries([template_, transformed_source_])
    
def scaleRegistration(srcPath, targetPath, outPath, start_scale = 1.0, end_scale = 3.0, scale_step_width=0.1):
    
    startScale = start_scale
    endScale = end_scale
    scaleStepWidth = scale_step_width

    assert startScale < endScale

    srcCloud = np.loadtxt(srcPath).astype(np.float32)
    targetCloud = np.loadtxt(targetPath).astype(np.float32)

    currentScale = startScale
    best_result = sys.float_info.max
    best_iteration = -1
    current_iteration = 0
    best_transformation = None
    best_scale = -1
    while currentScale <= endScale:
        current_error, current_transformation = robustICP(srcPath, targetPath, outPath, currentScale, srcCloud, targetCloud)
        if current_error < best_result:
            best_result = current_error
            best_iteration = current_iteration
            best_transformation = current_transformation
            best_scale = currentScale

        #update scale
        currentScale = currentScale + scaleStepWidth
        current_iteration += 1

    # DEBUG
    srcCloud = np.loadtxt(srcPath).astype(np.float32)
    srcCloud *= best_scale
    transformedSrc = np.hstack((srcCloud, np.ones((srcCloud.shape[0], 1))))  #(nx3)->(nx4)
    transformedSrc = transformedSrc.dot(best_transformation.T)[:, 0:3]
    targetCloud = np.loadtxt(targetPath).astype(np.float32)
    display_open3d(targetCloud, srcCloud, transformedSrc)

    return best_transformation, best_scale