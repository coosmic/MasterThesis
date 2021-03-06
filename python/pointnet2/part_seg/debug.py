import open3d as o3d
import numpy as np
import random

def showCloudTransformation(src, tgt, rotation, translation, showOrgSrc = False):
    pcdTgt = o3d.geometry.PointCloud()
    pcdTgt.points = o3d.utility.Vector3dVector(tgt.T)
    pcdTgt.paint_uniform_color([0, 1, 0])

    print(src.shape)
    print(rotation.shape)

    pcdSrc = o3d.geometry.PointCloud()
    pcdSrc.points = o3d.utility.Vector3dVector(src.T)
    pcdSrc.paint_uniform_color([0, 0, 1])

    srctmp = src.T.dot(rotation)
    #print(srctmp)
    #print(translation_ab_pred[0].cpu().detach().numpy())
    srctmp += translation
    #print(srctmp)
    pcdSrcT = o3d.geometry.PointCloud()
    pcdSrcT.points = o3d.utility.Vector3dVector(srctmp)
    pcdSrcT.paint_uniform_color([1, 0, 0])

    if showOrgSrc:
        o3d.visualization.draw_geometries([pcdSrc, pcdSrcT, pcdTgt])
    else:
        o3d.visualization.draw_geometries([pcdSrcT, pcdTgt])


def showClouds(src, tgt, other = []):
    pcdTgt = o3d.geometry.PointCloud()
    pcdTgt.points = o3d.utility.Vector3dVector(tgt.T)
    pcdTgt.paint_uniform_color([0, 1, 0])

    pcdSrc = o3d.geometry.PointCloud()
    pcdSrc.points = o3d.utility.Vector3dVector(src.T)
    pcdSrc.paint_uniform_color([1, 0, 0])

    allPointClouds = []
    allPointClouds.append(pcdSrc)
    allPointClouds.append(pcdTgt)

    for nparr in other:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(nparr.T)
        pcd.paint_uniform_color([0, 0, 1])
        allPointClouds.append(pcd)

    o3d.visualization.draw_geometries(allPointClouds)

def showSegmentation(points, predictions):

    print(points.shape)
    print(predictions.shape)
    print(predictions)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if(pcd.has_colors()):
        print("has color")
    else:
        print("no color")
        pcd.paint_uniform_color([1, 1, 1])

    for i in range(len(pcd.colors)):
        #print(predictions[i])
        if predictions[i] == 0:
            pcd.colors[i] = [1, 0, 0]
            continue
        if predictions[i] == 1:
            pcd.colors[i] = [0, 1, 0]
            continue
        if predictions[i] == 2:
            pcd.colors[i] = [0, 0, 1]
            continue
        #print(pcd.colors[i])

    o3d.visualization.draw_geometries([pcd])

def showColoredLeaves(leaves):
    allLeavePointClouds = []
    for leavePoints in leaves:
        print(leavePoints.shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(leavePoints[:,0:3])
        pcd.paint_uniform_color([random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)])
        allLeavePointClouds.append(pcd)
    o3d.visualization.draw_geometries(allLeavePointClouds)
