import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
import os
import psutil
import random
import json

def createResultObject(path):
    result = {
        "LeaveCount" : -1,
        "Height" : -1,
        "Volume" : -1,
        "GrothSinceLastSnapshot" : -1,
        "BackgroundRegistration" : {"Transformation" : "", "Scale" : -1}
    }

    with open(os.path.join(path, "result.json"), 'w') as f:
        json.dump(result, f)

def updateResultObject(path, resultUpdate):
    if not os.path.isfile(os.path.join(path, "result.json")):
       createResultObject(path)
    with open(os.path.join(path, "result.json"), 'r') as f:
        result = json.load(f)
        print(resultUpdate)
        if resultUpdate["JobName"] in result:
            result[resultUpdate["JobName"]] = resultUpdate["Value"]
        with open(os.path.join(path, "result.json"), 'w') as fs:
            json.dump(result, fs)

def getResultObject(path):
    if not os.path.isfile(os.path.join(path, "result.json")):
        raise Exception("Missing Result File for " + path)
    with open(os.path.join(path, "result.json"), 'r') as f:
        return json.load(f)

def predictionToColor(points, predictions, path):
    #print("points.shape ",points.shape)
    #print("predictions.shape ",predictions.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
    pcd.normals = o3d.utility.Vector3dVector(points[:, 3:6])
    pcd.paint_uniform_color([1, 1, 1])

    f = open(path, "w")

    assert(len(pcd.colors) == len(predictions))
    for i in range(len(pcd.colors)):
        line = str(points[ i, 0])+" "+str(points[i, 1])+" "+str(points[ i, 2])+" "+str(points[ i, 3])+" "+str(points[ i, 4])+" "+str(points[i, 5])
        #print(line)
        #print(predictions[i])
        if predictions[i] == 0:
            pcd.colors[i] = [0, 0, 1]
            #line = line + " 0 0 1\n"
            #f.write(line)
            continue
        if predictions[i] == 1:
            pcd.colors[i] = [0, 1, 0]
            #line = line + " 0 1 0\n"
            #f.write(line)
            continue
        if predictions[i] == 2:
            pcd.colors[i] = [1, 0, 0]
            #line = line + " 1 0 0\n"
            #f.write(line)
            continue
        
    #f.close()
    o3d.io.write_point_cloud(path, pcd, False, True, False)
    
def improveBackgroundPrediction(points, predictions):
    kdtree=KDTree(points[:, 0:3])
    dist,neighborIndices=kdtree.query(points[:, 0:3], 10)

    changesOccured = True
    maxIterCount = 0 #Prevent infinit looping
    while changesOccured and maxIterCount < 10:
        changesOccured = False
        #print("improve Background prediction loop")
        for i in range(len(points)):
            neighborPredictions = predictions[neighborIndices[i]]
            #print(neighborPredictions.shape)
            #print(neighborPredictions)
            y = np.bincount(neighborPredictions)
            #print(y)
            ii = np.nonzero(y)[0]
            #print(ii)
            zib = list(zip(ii,y[ii]))

            for j in range(len(zib)):
                if(zib[j][0] == 2):
                    #zib[j][1] = zib[j][1] * 0.5
                    lst = list(zib[j])
                    lst[1] = lst[1] * 0.5
                    zib[j] = tuple(lst)
                if(zib[j][0] == 1):
                    lst = list(zib[j])
                    lst[1] = lst[1] * 1.5
                    zib[j] = tuple(lst)

            zib.sort(key=lambda x:x[1], reverse=True)


            bestScore = zib[0][0]
            if(predictions[i] != bestScore):
                #print("Neighbor Prediction Counts (sorted): ", zib)
                #print("Neighbor Predictions: ", neighborPredictions)
                #print("Original Point Prediction: ", predictions[i])
                predictions[i] = bestScore
                changesOccured = True
        maxIterCount +=1 
    return predictions

def floodFill3D(points, distance=0.0075, startIndex=0):
    kdtree=KDTree(points[:, 0:3])

    #remainingIndicies = np.arange(0, len(points), 1)
    foundIndicies = []
    currentIndices = [startIndex]
    while len(currentIndices) > 0:
        dist,neighborIndices = kdtree.query(points[currentIndices[0], 0:3], k=len(points), p=2, distance_upper_bound=distance, workers=psutil.cpu_count(logical=False))
        #print("dist.len", len(dist))
        #print("neighborIndices.len", len(neighborIndices))
        #print(len(dist[neighborIndices] <= distance))
        filterRes = dist <= distance
        #print(filterRes)
        distanceFilteredNeighborIndices = neighborIndices[filterRes]
        filteredNeighborIndices = distanceFilteredNeighborIndices[np.in1d(distanceFilteredNeighborIndices[:], foundIndicies, invert=True)]

        currentIndices = np.delete(currentIndices, 0, 0)
        currentIndices = np.append(currentIndices, filteredNeighborIndices)
        foundIndicies = np.append(foundIndicies, filteredNeighborIndices)

    #print("found indices: ",foundIndicies)
    #print("found indices len: ",len(foundIndicies))
    return foundIndicies.astype(int)

def saveLeavePredictions(leaves, path):
    pcd_combined = o3d.geometry.PointCloud()
    for leavePoints in leaves:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(leavePoints[:,0:3])
        pcd.paint_uniform_color([random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)])
        pcd_combined += pcd
    o3d.io.write_point_cloud(path, pcd_combined, False, True, False)