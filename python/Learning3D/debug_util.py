import open3d as o3d
import numpy as np


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
    srctmp += translation
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