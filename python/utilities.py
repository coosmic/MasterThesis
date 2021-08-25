import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
import os

def getPipelineStatus(folderPath):
    pipelineStatus  = {
        "ImagesUploaded" : False,
        "PointCloudGenerated" : False,
        "ShapenetFormat" : False,
        "BackgroundSegmented" : False,
        "LeavesSegmented" : False
    }
    print(folderPath)
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
                print(os.path.join(currentDirPath,t))
                if t != "background":
                    state[d][t] = getPipelineStatus(os.path.join(currentDirPath,t))
                else:
                    state[d][t] = getPipelineStatusBackground(os.path.join(currentDirPath,t))
    return state

def updateState(currentState, stateUpdate):
    dataSet = stateUpdate["jobParameter"]["testSet"]
    timeStamp = stateUpdate["jobParameter"]["timeStamp"]

    if dataSet not in currentState:
        currentState[dataSet] = {}
        if timeStamp not in currentState[dataSet]:
            if timeStamp != "background":
                currentState[dataSet][timeStamp] = {
                    "ImagesUploaded" : False,
                    "PointCloudGenerated" : False,
                    "ShapenetFormat" : False,
                    "BackgroundSegmented" : False,
                    "LeavesSegmented" : False
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

    return currentState

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