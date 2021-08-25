from pointnet2.part_seg import estimate, estimate2, evaluate2
import os
import utilities
import numpy as np
import json
import psutil

PARTSEG2 = None
PARTSEG3 = None


def startPartSegPipelines():
    global PARTSEG2
    PARTSEG2 = estimate.Pointnet2PartSegmentation2("pointnet2_part_seg", "./pointnet2/part_seg/results/t6_2ClassesPartSeg/model.ckpt")
    #PARTSEG2 = estimate.Pointnet2PartSegmentation("pointnet2_part_seg3C", "./pointnet2/part_seg/results/t5_3ClassesPartSeg3C/model.ckpt", {'Plant': [0, 1, 2] })
    global PARTSEG3
    PARTSEG3 = estimate.Pointnet2PartSegmentation2("pointnet2_part_seg3C", "./pointnet2/part_seg/results/t5_3ClassesPartSeg3C/model.ckpt", {'Plant': [0, 1, 2] })

def jobGeneratePointCloud(jobParameter):
    cpus = psutil.cpu_count(logical=False)
    cpus = cpus - 2 # spare two cores for system and server
    if cpus < 1:
        cpus = 1
    makePointCloudCommand = f"docker run -ti --rm -v {jobParameter['baseFolderPath']}/{jobParameter['testSet']}:/{jobParameter['testSet']} --cpus {cpus} --gpus all opendronemap/odm:gpu --rerun-all -e  odm_filterpoints --project-path /{jobParameter['testSet']} {jobParameter['timeStamp']}"
    os.system(makePointCloudCommand)
    print("pc job started ", makePointCloudCommand)

def jobLeavesSegmentation(jobParameter):
    basePath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'] )
    pointCloudPath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'], 'shapenet', "CloudWithoutBackgroundSS1.txt" )
    if os.path.isfile(pointCloudPath):
        print("Leave Segmentation started")
        estimationLocation = os.path.join(os.path.abspath(os.getcwd()), 'data', 'predictions/leave/13371337/point_cloudSS1.txt' )
        copyCommand = f"cp {pointCloudPath} {estimationLocation}"
        os.system(copyCommand)

        evaluate2.evaluate()

        pointCloudPath = os.path.join(basePath, 'shapenet', "point_cloudSS1LeavePrediction.txt" )
        estimationLocation = os.path.join(os.path.abspath(os.getcwd()), 'data', 'predictions/leave/Plant/point_cloudSS1.txt' )
        copyCommand = f"cp {estimationLocation} {pointCloudPath}"
        os.system(copyCommand)

        predictionPath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'], 'shapenet', "point_cloudSS1LeavePrediction.pcd" )
        data = np.loadtxt(pointCloudPath).astype(np.float32)
        #print("data.shape ", data.shape)
        utilities.predictionToColor(data[:, 0:6], data[:, 6:], predictionPath)

        #sampledCloud, predictions = PARTSEG3.estimate(pointCloudPath)
        #predictions = utilities.improveBackgroundPrediction(sampledCloud, predictions)
        #predictionPath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'], 'shapenet', "point_cloudSS1LeavePrediction.pcd" )
        #utilities.predictionToColor(sampledCloud, predictions, predictionPath)
    else:
        print(f"File {pointCloudPath} is not existing")
        #TODO Error handling

def jobBackgroundSegmentation(jobParameter):
    basePath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'] )
    pointCloudPath = os.path.join(basePath, 'shapenet', "point_cloudSS1.txt" )
    if os.path.isfile(pointCloudPath):
        print("Background Segmentation started")
        estimationLocation = os.path.join(os.path.abspath(os.getcwd()), 'data', 'predictions/background/13371337/point_cloudSS1.txt' )
        copyCommand = f"cp {pointCloudPath} {estimationLocation}"
        os.system(copyCommand)

        PARTSEG3.predict()

        pointCloudPath = os.path.join(basePath, 'shapenet', "point_cloudSS1BackgroundPrediction.txt" )
        estimationLocation = os.path.join(os.path.abspath(os.getcwd()), 'data', 'predictions/background/Plant/point_cloudSS1.txt' )
        copyCommand = f"cp {estimationLocation} {pointCloudPath}"
        os.system(copyCommand)

        predictionPath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'], 'shapenet', "point_cloudSS1BackgroundPrediction.pcd" )
        data = np.loadtxt(pointCloudPath).astype(np.float32)
        #print("data.shape ", data.shape)
        #print(data)
        #print(data[:, 0:6])
        #print(data[:, 6:])
        predictions = utilities.improveBackgroundPrediction(data[:, 0:6], data[:, 6].astype(np.int))
        utilities.predictionToColor(data[:, 0:6], predictions, predictionPath)

        orgCloudPath = os.path.join(basePath, 'shapenet', "point_cloud.ply" )
        outFolder = os.path.join(basePath, 'shapenet')
        removeBackgroundCommand = f"../build/pgm -J BackgroundRemovalPipeline --SourceCloudPath {predictionPath} --TargetCloudPath {orgCloudPath} --OutputFolder {outFolder} --SearchRadius 0.0075"
        #print(removeBackgroundCommand)
        os.system(removeBackgroundCommand)
        toShapenetFormatCommand = f"../build/pgm -J Shapenet --snin {os.path.join(basePath, 'shapenet', 'CloudWithoutBackground.ply')} --snout {os.path.join(basePath, 'shapenet', 'CloudWithoutBackground')} --RemoveBackground false --MaxSubsample 1"
        #print(toShapenetFormatCommand)
        os.system(toShapenetFormatCommand)

        #estimationLocation = os.path.join(os.path.abspath(os.getcwd()), 'data', 'predictions/background/13371337/point_cloudSS1.txt' )
        #copyCommand = f"cp {pointCloudPath} {estimationLocation}"
        #os.system(copyCommand)
        #estimate2.predict()

        #sampledCloud, predictions  = PARTSEG3.startPointnet2PartSegmentation(pointCloudPath)
        #predictions = utilities.improveBackgroundPrediction(sampledCloud, predictions)
        #predictionPath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'], 'shapenet', "point_cloudSS1BackgroundPrediction.pcd" )
        #data = np.loadtxt(pointCloudPath).astype(np.float32)
        #utilities.predictionToColor(data, predictions, predictionPath)

        #orgCloudPath = os.path.join(basePath, 'shapenet', "point_cloud.ply" )
        #outFolder = os.path.join(basePath, 'shapenet')
        #removeBackgroundCommand = f"../build/pgm -J BackgroundRemovalPipeline --SourceCloudPath {predictionPath} --TargetCloudPath {orgCloudPath} --OutputFolder {outFolder} --SearchRadius 0.0075"
        #print(removeBackgroundCommand)
        #os.system(removeBackgroundCommand)
        #toShapenetFormatCommand = f"../build/pgm -J Shapenet --snin {os.path.join(basePath, 'shapenet', 'CloudWithoutBackground.ply')} --snout {os.path.join(basePath, 'shapenet', 'CloudWithoutBackground')} --RemoveBackground false --MaxSubsample 1"
        #print(toShapenetFormatCommand)
        #os.system(toShapenetFormatCommand)
        
    else:
        print(f"File {pointCloudPath} is not existing")
        #TODO Error handling

def jobToShapenetFormat(jobParameter):
    print("convert to shapenet started")
    basePath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'] )
    if not os.path.isdir(os.path.join(basePath, 'shapenet')):
        os.makedirs(os.path.join(basePath, 'shapenet'))
    toShapenetFormatCommand = f"../build/pgm -J Shapenet --snin {os.path.join(basePath, 'odm_filterpoints', 'point_cloud.ply')} --snout {os.path.join(basePath, 'shapenet', 'point_cloud')} --RemoveBackground false --MaxSubsample 1"
    
    print(toShapenetFormatCommand)
    os.system(toShapenetFormatCommand)
    pass



knownJobs = {
        "ConvertToShapenet" : jobToShapenetFormat,
        "SegmentLeaves" : jobLeavesSegmentation,
        "SegmentBackground" : jobBackgroundSegmentation
    }
def genericJob(jobParameter):
    jobName = jobParameter["jobName"]
    
    jobParams = jobParameter["jobParameter"]
    jobParams["testSet"] = jobParameter["testSet"]
    jobParams["timeStamp"] = jobParameter["timeStamp"]

    if jobName in knownJobs:
        print(f"{jobName} will be started")
        knownJobs[jobName](jobParams)

def getJob(jobName, testSet, timeStamp):

    if jobName not in knownJobs:
        return None

    jobDefintion = {"job" : knownJobs[jobName], "jobName" : jobName, "data" : {"testSet" : testSet, "timeStamp" : timeStamp}}

    return jobDefintion