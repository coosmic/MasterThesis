import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
import os
import psutil
import random
import json

def getPipelineStatus(folderPath):
    pipelineStatus  = {
        "ImagesUploaded" : False,
        "PointCloudGenerated" : False,
        "ShapenetFormat" : False,
        "BackgroundSegmented" : False,
        "LeavesSegmented" : False,
        "LeaveStemSplit" : False,
        "CountLeaves" : False,
        "BackgroundRegistration" : False
    }
    #print(folderPath)
    if os.path.isdir(os.path.join(folderPath, "images")) and len(os.listdir(os.path.join( folderPath, "images"))) != 0:
        pipelineStatus["ImagesUploaded"] = True
    if os.path.isfile(os.path.join( folderPath, "odm_filterpoints", "point_cloud.ply")):
        pipelineStatus["PointCloudGenerated"] = True
    if not os.path.isdir(os.path.join( folderPath, "shapenet") ):
        return pipelineStatus
    if os.path.isfile(os.path.join( folderPath, "shapenet", "point_cloudSS1.txt")):
        pipelineStatus["ShapenetFormat"] = True
    if os.path.isfile(os.path.join( folderPath, "shapenet", "point_cloudSS1BackgroundPrediction.pcd")):
        pipelineStatus["BackgroundSegmented"] = True
    if os.path.isfile(os.path.join( folderPath, "shapenet", "point_cloudSS1LeavePrediction.pcd")):
        pipelineStatus["LeavesSegmented"] = True
    if os.path.isfile(os.path.join( folderPath, "shapenet", "stem.txt")) and os.path.isfile(os.path.join( folderPath, "shapenet", "leaves.txt")):
        pipelineStatus["LeaveStemSplit"] = True
    if os.path.isfile(os.path.join( folderPath, "shapenet", "leavesSegmented.pcd")):
        pipelineStatus["CountLeaves"] = True
    return pipelineStatus

def getPipelineStatusBackground(folderPath):
    pipelineStatus  = {
        "ImagesUploaded" : False,
        "PointCloudGenerated" : False,
        "ShapenetFormat" : False
    }

    if os.path.isdir(os.path.join( folderPath, "images")) and len(os.listdir(os.path.join( folderPath, "images"))) != 0:
        pipelineStatus["ImagesUploaded"] = True
    if os.path.isfile(os.path.join( folderPath, "odm_filterpoints", "point_cloud.ply")):
        pipelineStatus["PointCloudGenerated"] = True
    if os.path.isfile(os.path.join( folderPath, "shapenet", "point_cloudSS1.txt")):
        pipelineStatus["ShapenetFormat"] = True
    return pipelineStatus

def restoreState(dataPath):
    state = {}

    dirs = [name for name in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath,name))]
    for d in dirs:
        if d != "predictions":
            currentDirPath = os.path.join(dataPath,d)
            currentTimestamps = dirs = [name for name in os.listdir(currentDirPath) if os.path.isdir(os.path.join(currentDirPath,name))]
            state[d] = {}
            for t in currentTimestamps:
                #print(os.path.join(currentDirPath,t))
                if t != "background":
                    state[d][t] = getPipelineStatus(os.path.join(currentDirPath,t))
                else:
                    state[d][t] = getPipelineStatusBackground(os.path.join(currentDirPath,t))
    return state

def updateState(currentState, stateUpdate):
    dataSet = stateUpdate["data"]["testSet"]
    timeStamp = stateUpdate["data"]["timeStamp"]

    if dataSet not in currentState:
        currentState[dataSet] = {}
        if timeStamp not in currentState[dataSet]:
            if timeStamp != "background":
                currentState[dataSet][timeStamp] = {
                    "ImagesUploaded" : False,
                    "PointCloudGenerated" : False,
                    "ShapenetFormat" : False,
                    "BackgroundSegmented" : False,
                    "LeavesSegmented" : False,
                    "LeaveStemSplit" : False,
                    "CountLeaves" : False,
                    "BackgroundRegistration" : False
                }
            else:
                currentState[dataSet][timeStamp] = {
                    "ImagesUploaded" : False,
                    "PointCloudGenerated" : False,
                    "ShapenetFormat" : False
                }
    if stateUpdate["jobName"] == "SaveImages":
        currentState[dataSet][timeStamp]["ImagesUploaded"] = True
    if stateUpdate["jobName"] == "GeneratePointCloud":
        currentState[dataSet][timeStamp]["PointCloudGenerated"] = True
    if stateUpdate["jobName"] == "ConvertToShapenet":
        currentState[dataSet][timeStamp]["ShapenetFormat"] = True
    if stateUpdate["jobName"] == "SegmentLeaves":
        currentState[dataSet][timeStamp]["LeavesSegmented"] = True
    if stateUpdate["jobName"] == "SegmentBackground":
        currentState[dataSet][timeStamp]["BackgroundSegmented"] = True
    if stateUpdate["jobName"] == "LeaveStemSplit":
        currentState[dataSet][timeStamp]["LeaveStemSplit"] = True
    if stateUpdate["jobName"] == "CountLeaves":
        currentState[dataSet][timeStamp]["CountLeaves"] = True
    if stateUpdate["jobName"] == "BackgroundRegistration":
        currentState[dataSet][timeStamp]["BackgroundRegistration"] = True

    return currentState

def createResultObject(path):
    result = {
        "LeaveCount" : -1,
        "Height" : -1,
        "Volume" : -1,
        "grothSinceLastSnapshot" : -1
    }

    with open(os.path.join(path, "result.json"), 'w') as f:
        json.dump(result, f)



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