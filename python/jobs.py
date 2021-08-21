from pointnet2.part_seg import estimate
import os

PARTSEG2 = None
PARTSEG3 = None


def startPartSegPipelines():
    global PARTSEG2
    PARTSEG2 = estimate.Pointnet2PartSegmentation("pointnet2_part_seg", "./pointnet2/part_seg/results/t3_2ClassesPartSeg_1024_256/model.ckpt")
    global PARTSEG3
    PARTSEG3 = estimate.Pointnet2PartSegmentation("pointnet2_part_seg3C", "./pointnet2/part_seg/results/t4_3ClassesPartSeg/model.ckpt", {'Plant': [0, 1, 2] })

def jobGeneratePointCloud(jobParameter):
    makePointCloudCommand = f"docker run -ti --rm -v {jobParameter['baseFolderPath']}/{jobParameter['testSet']}:/{jobParameter['testSet']} --gpus all opendronemap/odm:gpu --rerun-all -e  odm_filterpoints --project-path /{jobParameter['testSet']} {jobParameter['timeStamp']}"
    #os.system(makePointCloudCommand)
    print("pc job started ", makePointCloudCommand)

def jobLeavesSegmentation(jobParameter):
    pointCloudPath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'], 'shapenet', "point_cloudSS1.txt" )
    if os.path.isfile(pointCloudPath):
        print("Leave Segmentation started")
        result = PARTSEG2.estimate(pointCloudPath)
    else:
        print(f"File {pointCloudPath} is not existing")
        #TODO Error handling

def jobBackgroundSegmentation(jobParameter):
    pointCloudPath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'], 'shapenet', "point_cloudSS1.txt" )
    if os.path.isfile(pointCloudPath):
        print("Leave Segmentation started")
        result = PARTSEG3.estimate(pointCloudPath)
    else:
        print(f"File {pointCloudPath} is not existing")
        #TODO Error handling

def jobToShapenetFormat(jobParameter):
    print("convert to shapenet started")
    basePath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'] )
    if not os.path.isdir(os.path.join(basePath, 'shapenet')):
        os.makedirs(os.path.join(basePath, 'shapenet'))
    toShapenetFormatCommand = f"../build/pgm -J Shapenet --snin {os.path.join(basePath, 'odm_filterpoints', 'point_cloud.ply')} --snout {os.path.join(basePath, 'shapenet', 'point_cloud')} --RemoveBackground false"
    print(toShapenetFormatCommand)
    os.system(toShapenetFormatCommand)
    pass



knownJobs = {
        "convertToShapenet" : jobToShapenetFormat,
        "segmentLeaves" : jobLeavesSegmentation,
        "segmentBackground" : jobBackgroundSegmentation
    }
def genericJob(jobParameter):
    jobName = jobParameter["jobName"]
    
    jobParams = jobParameter["jobParameter"]
    jobParams["testSet"] = jobParameter["testSet"]
    jobParams["timeStamp"] = jobParameter["timeStamp"]

    if jobName in knownJobs:
        print(f"{jobName} will be started")
        knownJobs[jobName](jobParams)