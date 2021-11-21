from pointnet2.part_seg import estimate, evaluate2, debug
import os
import utilities
import numpy as np
import psutil
import constants
import time

#Used for RICP
import registration

from Learning3D.registration import runWithDifferentScales, loadRawDataWithoutArgs, loadModel, loadDataLoader

PARTSEG2 = None
PARTSEG3 = None


def startPartSegPipelines():
    global PARTSEG3
    PARTSEG3 = estimate.Pointnet2PartSegmentation2("pointnet2_part_seg3C", "./pointnet2/part_seg/results/training/t7_3ClassesPartSegNoNorm/model.ckpt", {'Plant': [0, 1, 2] })

def jobGeneratePointCloud(jobParameter):
    cpus = psutil.cpu_count(logical=False)
    cpus = cpus - 2 # spare two cores for system and server
    if cpus < 1:
        cpus = 1
    makePointCloudCommand = f'docker run -ti --rm --user "$(id -u):$(id -g)" -v {os.path.join(os.path.abspath(os.getcwd()), "data" )}/{jobParameter["testSet"]}:/{jobParameter["testSet"]} --cpus {cpus} --gpus all opendronemap/odm:gpu --rerun-all -e  odm_filterpoints --project-path /{jobParameter["testSet"]} {jobParameter["timeStamp"]}'
    print("starting ", makePointCloudCommand)
    os.system(makePointCloudCommand)

    #cleanup
    basePath = f'{os.path.join(os.path.abspath(os.getcwd()), "data" )}/{jobParameter["testSet"]}/{jobParameter["timeStamp"]}'
    cleanupCommand = f'rm -r {basePath}/odm_georeferencing ; rm -r {basePath}/opensfm ; rm {basePath}/cameras.json ; rm {basePath}/images.json ; rm {basePath}/img_list.txt'
    os.system(cleanupCommand)

    return {"Status": constants.statusDone}


def jobLeavesSegmentation(jobParameter):
    basePath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'] )
    pointCloudPath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'], 'shapenet', "CloudWithoutBackgroundSS1.txt" )
    if os.path.isfile(pointCloudPath):
        print("Leave Segmentation started")
        estimationLocation = os.path.join(os.path.abspath(os.getcwd()), 'data', 'predictions/background/13371337/point_cloudSS1.txt' )
        copyCommand = f"cp {pointCloudPath} {estimationLocation}"
        os.system(copyCommand)

        evaluate2.evaluate()

        pointCloudPath = os.path.join(basePath, 'shapenet', "point_cloudSS1LeavePrediction.txt" )
        estimationLocation = os.path.join(os.path.abspath(os.getcwd()), 'data', 'predictions/background/Plant/point_cloudSS1.txt' )
        copyCommand = f"cp {estimationLocation} {pointCloudPath}"
        os.system(copyCommand)

        predictionPath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'], 'shapenet', "point_cloudSS1LeavePrediction.pcd" )
        data = np.loadtxt(pointCloudPath).astype(np.float32)
        predictions = utilities.improveBackgroundPrediction(data[:, 0:6], data[:, 6].astype(np.int))
        utilities.predictionToColor(data[:, 0:6], predictions, predictionPath)
        
        return {"Status": constants.statusDone}
    else:
        print(f"File {pointCloudPath} is not existing")
        #TODO Error handling
        return {"Status": constants.statusFailed, "Reason": f"File {pointCloudPath} is not existing"}

def jobBackgroundSegmentation(jobParameter):
    print(f"started background segmentation for {jobParameter['testSet']}/{jobParameter['timeStamp']}")
    basePath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'] )
    pointCloudPath = os.path.join(basePath, 'shapenet', "point_cloudSS1.txt" )
    if os.path.isfile(pointCloudPath):
        print("Background Segmentation started")
        estimationLocation = os.path.join(os.path.abspath(os.getcwd()), 'data', 'predictions/background/13371337/point_cloudSS1.txt' )
        copyCommand = f"cp {pointCloudPath} {estimationLocation}"
        print(copyCommand)
        os.system(copyCommand)
        time.sleep(0.25)

        PARTSEG3.predict()

        pointCloudPath = os.path.join(basePath, 'shapenet', "point_cloudSS1BackgroundPrediction.txt" )
        estimationLocation = os.path.join(os.path.abspath(os.getcwd()), 'data', 'predictions/background/Plant/point_cloudSS1.txt' )
        copyCommand = f"cp {estimationLocation} {pointCloudPath}"
        os.system(copyCommand)
        print(copyCommand)
        time.sleep(0.25)

        predictionPath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'], 'shapenet', "point_cloudSS1BackgroundPrediction.pcd" )
        data = np.loadtxt(pointCloudPath).astype(np.float32)
        predictions = utilities.improveBackgroundPrediction(data[:, 0:6], data[:, 6].astype(np.int))
        utilities.predictionToColor(data[:, 0:6], predictions, predictionPath)

        orgCloudPath = os.path.join(basePath, 'shapenet', "point_cloud.ply" )
        outFolder = os.path.join(basePath, 'shapenet')
        removeBackgroundCommand = f"../build/pgm -J BackgroundRemovalPipeline --SourceCloudPath {predictionPath} --TargetCloudPath {orgCloudPath} --OutputFolder {outFolder} --SearchRadius 0.0075"
        os.system(removeBackgroundCommand)


        toShapenetFormatCommand = f"../build/pgm -J Shapenet --in {os.path.join(basePath, 'shapenet', 'CloudWithoutBackground.ply')} --out {os.path.join(basePath, 'shapenet', 'CloudWithoutBackground')} --RemoveBackground false --MaxSubsample 1 --NoPlaneAlignment"
        os.system(toShapenetFormatCommand)
        return {"Status": constants.statusDone}
        
    else:
        print(f"File {pointCloudPath} is not existing")
        #TODO Error handling
        return {"Status": constants.statusFailed, "Reason": f"File {pointCloudPath} is not existing"}

def jobToShapenetFormat(jobParameter):
    print("convert to shapenet started")
    basePath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'] )
    if not os.path.isdir(os.path.join(basePath, 'shapenet')):
        os.makedirs(os.path.join(basePath, 'shapenet'))
    toShapenetFormatCommand = f"../build/pgm -J Shapenet --in {os.path.join(basePath, 'odm_filterpoints', 'point_cloud.ply')} --out {os.path.join(basePath, 'shapenet', 'point_cloud')} --RemoveBackground false --MaxSubsample 1"
    
    print(toShapenetFormatCommand)
    os.system(toShapenetFormatCommand)
    return {"Status": constants.statusDone}

def jobLeaveStemSplit(jobParameter):
    pointCloudPath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'], "shapenet", "point_cloudSS1LeavePrediction.txt")
    rawData = np.loadtxt(pointCloudPath).astype(np.float32)
    points = rawData[:, 0:6]
    predictions = rawData[:, 6]

    leavesPath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'], "shapenet", "leaves.txt")
    stemPath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'], "shapenet", "stem.txt")

    stemPoints = points[predictions == 1]
    leavePoints = points[predictions == 0]

    np.savetxt(stemPath, stemPoints)
    np.savetxt(leavesPath, leavePoints)
    return {"Status": constants.statusDone}

def jobCountLeaves(jobParameter):
    pointCloudPath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'], "shapenet", "leaves.txt")
    points = np.loadtxt(pointCloudPath).astype(np.float32)

    leaveCount = 0
    leavePoints = []
    distance = 0.025
    if "distance" in jobParameter:
        distance = jobParameter["distance"]
    print("Searching with distance", distance)
    while len(points > 0):
        selectedIndices = utilities.floodFill3D(points, distance)
        currentPoints = points[selectedIndices]
        if len(currentPoints) > 10:
            leavePoints.append(currentPoints)
            leaveCount += 1
        points = np.delete(points, selectedIndices, 0)
    print(f"Found {leaveCount} leaves")
    utilities.saveLeavePredictions(leavePoints, os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'], "shapenet", "leavesSegmented.pcd"))

    #resultsPath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'], "results.json")

    return {"Status": constants.statusDone, "Results": {"JobName": "LeaveCount", "Value": leaveCount}}

def jobBackgroundRegistration(jobParameter):
    folderPath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'])
    srcPath = os.path.join(folderPath, 'shapenet', 'registrationFormat.txt')
    targetPath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], 'background', 'shapenet', 'registrationFormat.txt')
    outPath = os.path.join(folderPath, 'shapenet', 'registration/')

    showResults = False
    if 'ShowResult' in jobParameter:
        showResults = jobParameter['ShowResult']

    registrationMethod = 'ICP'
    if 'RegistrationMethod' in jobParameter:
        registrationMethod = jobParameter['RegistrationMethod']

    netName = 'PointNetLK'
    runWithICP = True
    if registrationMethod == 'RICP':
        transformation, scale = registration.scaleRegistration(srcPath, targetPath, outPath)   #RICP
        return {"Status": constants.statusDone, "Results" : {"JobName": "BackgroundRegistration", "Value": {"Transformation" : transformation.tolist(), "Scale" : scale}} }
    elif registrationMethod == 'ICP':
        model = None #ICP
        data = loadRawDataWithoutArgs(srcPath, targetPath, 2048) # For ICP
    elif registrationMethod == 'PointNetLK':
        runWithICP = False
        model = loadModel("PointNetLK")
        data = loadDataLoader(srcPath, targetPath, 2048) # For Nets
    elif registrationMethod == 'DCP':
        runWithICP = False
        netName = 'DCP'
        model = loadModel("DCP")
        data = loadDataLoader(srcPath, targetPath, 2048) # For Nets
    elif registrationMethod == 'RPM':
        runWithICP = False
        netName = 'RPM'
        model = loadModel("RPM")
        data = loadDataLoader(srcPath, targetPath, 2048) # For Nets
    else:
        return {"Status": constants.statusFailed, "Reason": f"Unknown  Registration Method {registrationMethod}"}

    transformation, scale = runWithDifferentScales(data, show=showResults, model=model, use_icp=runWithICP, net=netName)

    print("Transformation", transformation)
    print("Scale", scale)

    return {"Status": constants.statusDone, "Results" : {"JobName": "BackgroundRegistration", "Value": {"Transformation" : transformation.tolist(), "Scale" : scale}} }

def jobConvertToBackground(jobParameter):
    folderPath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], jobParameter['timeStamp'])
    
    inPath = os.path.join(folderPath, 'shapenet', "CloudWithoutPlant.ply")
    if not os.path.exists(inPath):
        inPath = os.path.join(folderPath, 'odm_filterpoints', "point_cloud.ply")
    outPath = os.path.join(folderPath, 'shapenet', "registrationFormat.txt")

    if not os.path.isdir(os.path.join(folderPath, 'shapenet')):
        os.makedirs(os.path.join(folderPath, 'shapenet'))

    centerOnly = "true"
    if 'ExtractCenter' in jobParameter:
        if not jobParameter['ExtractCenter']:
            centerOnly = "false"
    useShapenet = "true"
    if 'UseShapenetFormat' in jobParameter:
        if not jobParameter['UseShapenetFormat']:
            useShapenet = "false"

    registrationFormatCommand = f"../build/pgm -J RegistrationFormat --in {inPath} --out {outPath} --SubsamplePointCount 2048 --CenterOnly "+centerOnly+" --UseShapenetFormat "+useShapenet
    os.system(registrationFormatCommand)
    return {"Status": constants.statusDone}

def jobCalculateSizes(jobParameter):
    timestamp = jobParameter['timeStamp']
    if timestamp == "background":
        return {"Status": constants.statusNotAllowed}
    
    baseFolderPath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'], timestamp)
    pathToFile = os.path.join(baseFolderPath, 'shapenet', 'point_cloudSS1BackgroundPrediction.txt')

    try:
        results = utilities.getResultObject(baseFolderPath)
    except Exception as e:
        return {"Status": constants.statusFailed, "Details": "Result is missing. Run BackgroundRegistration"}
    if "BackgroundRegistration" not in results:
        return {"Status": constants.statusFailed, "Details": "BackgroundRegistration Result is missing. Run BackgroundRegistration"}
    
    bgRegistrationResult = results['BackgroundRegistration']

    scale = bgRegistrationResult['Scale']
    if scale == -1:
        return {"Status": constants.statusFailed, "Details": "BackgroundRegistration Result is invalid. Run BackgroundRegistration"}

    transformation = np.asarray(bgRegistrationResult['Transformation'])
    
    pointsWithLabels = np.loadtxt(pathToFile).astype(np.float32)

    isBackgroundPoint = pointsWithLabels[:,6] != 2.0
    points = pointsWithLabels[isBackgroundPoint][:,0:3]
    print(points.shape)

    # Scale Point Cloud
    points = points * scale
    
    # Transform Point Cloud
    pointsTMP = np.hstack((points, np.ones((points.shape[0], 1))))  #(nx3)->(nx4)
    points_t = pointsTMP.dot(transformation.T)[:,:-1]

    print(points_t.shape)

    # Get Box Dimensions => Volume
    minX = np.min(points_t[:,0])
    maxX = np.max(points_t[:,0])

    minY = np.min(points_t[:,1])
    maxY = np.max(points_t[:,1])

    minZ = np.min(points_t[:,2])
    maxZ = np.max(points_t[:,2])

    lengthX = abs(minX) + maxX if minX < 0.0 else maxX - minX
    lengthY = abs(minY) + maxY if minY < 0.0 else maxY - minY
    lengthZ = abs(minZ) + maxZ if minZ < 0.0 else maxZ - minZ

    volume = lengthY * lengthX * lengthZ
    
    # Get Max Z => Height
    heigth = maxZ

    # Update Result
    utilities.updateResultObject(baseFolderPath, {'JobName': 'Height', 'Value' : heigth})
    utilities.updateResultObject(baseFolderPath, {'JobName': 'Volume', 'Value' : volume})

    return {"Status": constants.statusDone}

def jobCalculateGrowth(jobParameter):
    timestamp = jobParameter['timeStamp']
    if timestamp == "background":
        return {"Status": constants.statusNotAllowed}
    
    baseFolderPath = os.path.join(os.path.abspath(os.getcwd()), 'data', jobParameter['testSet'])
    if timestamp != "t1":
        lastTimestamp = "t"+str(int(timestamp.replace("t", ""))+1)

        try:
            resultsThis = utilities.getResultObject(os.path.join(baseFolderPath, timestamp))
        except Exception as e:
            return {"Status": constants.statusFailed, "Details": "Result for this missing. Readd images"}
        try:
            resultsLast = utilities.getResultObject(os.path.join(baseFolderPath, lastTimestamp))
        except Exception as e:
            return {"Status": constants.statusFailed, "Details": f"Result for last ({lastTimestamp}) missing."}
        if "Height" not in resultsThis:
            return {"Status": constants.statusFailed, "Details": f"Height Result is missing for {timestamp}. Run CalculateSize"}
        if "Height" not in resultsLast:
            return {"Status": constants.statusFailed, "Details": f"Height Result is missing for {lastTimestamp}. Run CalculateSize"}

        # get this height
        thisHeight = resultsThis["Height"]

        # get last height
        lastHeight = resultsLast["Height"]

        # calculate growth
        growth = thisHeight / lastHeight

        # add growth to result
        utilities.updateResultObject(os.path.join(baseFolderPath, timestamp), {'JobName': 'GrowthSinceLastSnapshot', 'Value' : growth})

        return {"Status": constants.statusDone}
    else:
        return {"Status": constants.statusNotAllowed, "Details": "Calculating growth for first timestamp is not possible"}

        


knownJobs = {
        "GeneratePointCloud" : jobGeneratePointCloud,
        "ConvertToShapenet" : jobToShapenetFormat,
        "SegmentLeaves" : jobLeavesSegmentation,
        "SegmentBackground" : jobBackgroundSegmentation,
        "LeaveStemSplit" : jobLeaveStemSplit,
        "CountLeaves" : jobCountLeaves,
        "BackgroundRegistration" : jobBackgroundRegistration,
        "ConvertToRegistration" : jobConvertToBackground,
        "CalculateSize" : jobCalculateSizes,
        "CalculateGrowth" : jobCalculateGrowth
}

def genericJob(jobParameter):
    jobName = jobParameter["jobName"]
    
    jobParams = jobParameter["jobParameter"]
    jobParams["testSet"] = jobParameter["testSet"]
    jobParams["timeStamp"] = jobParameter["timeStamp"]

    if jobName in knownJobs:
        print(f"{jobName} will be started")
        knownJobs[jobName](jobParams)

def getJob(jobName, testSet, timeStamp, parameter):

    if jobName not in knownJobs:
        return None

    parameter["testSet"] = testSet
    parameter["timeStamp"] = timeStamp

    jobDefintion = {"job" : knownJobs[jobName], "jobName" : jobName, "data" : parameter}

    return jobDefintion