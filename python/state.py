import os
import constants
import json
import utilities

class State():

    def __init__(self):
        self.state = {}

    def getPipelineStatus(self, folderPath):
        pipelineStatus  = {
            "ImagesUploaded" : {"Status" : constants.statusNotDone},
            "PointCloudGenerated" : {"Status" : constants.statusNotDone},
            "ShapenetFormat" : {"Status" : constants.statusNotDone},
            "BackgroundSegmented" : {"Status" : constants.statusNotDone},
            "LeavesSegmented" : {"Status" : constants.statusNotDone},
            "LeaveStemSplit" : {"Status" : constants.statusNotDone},
            "CountLeaves" : {"Status" : constants.statusNotDone},
            "BackgroundRegistration" : {"Status" : constants.statusNotDone},
            "ConvertToRegistration" : {"Status" : constants.statusNotDone},
            "CalculateSize" : {"Status" : constants.statusNotDone},
            "CalculateGrowth" : {"Status" : constants.statusNotDone}
        }
        
        if os.path.isdir(os.path.join(folderPath, "images")) and len(os.listdir(os.path.join( folderPath, "images"))) != 0:
            pipelineStatus["ImagesUploaded"] = {"Status" : constants.statusDone}
        if os.path.isfile(os.path.join( folderPath, "odm_filterpoints", "point_cloud.ply")):
            pipelineStatus["PointCloudGenerated"] = {"Status" : constants.statusDone}
        if not os.path.isdir(os.path.join( folderPath, "shapenet") ):
            return pipelineStatus
        if os.path.isfile(os.path.join( folderPath, "shapenet", "point_cloudSS1.txt")):
            pipelineStatus["ShapenetFormat"] = {"Status" : constants.statusDone}
        if os.path.isfile(os.path.join( folderPath, "shapenet", "point_cloudSS1BackgroundPrediction.pcd")):
            pipelineStatus["BackgroundSegmented"] = {"Status" : constants.statusDone}
        if os.path.isfile(os.path.join( folderPath, "shapenet", "point_cloudSS1LeavePrediction.pcd")):
            pipelineStatus["LeavesSegmented"] = {"Status" : constants.statusDone}
        if os.path.isfile(os.path.join( folderPath, "shapenet", "stem.txt")) and os.path.isfile(os.path.join( folderPath, "shapenet", "leaves.txt")):
            pipelineStatus["LeaveStemSplit"] = {"Status" : constants.statusDone}
        if os.path.isfile(os.path.join( folderPath, "shapenet", "leavesSegmented.pcd")):
            pipelineStatus["CountLeaves"] = {"Status" : constants.statusDone}
        if os.path.isfile(os.path.join(folderPath, 'shapenet', "registrationFormat.txt")):
            pipelineStatus["ConvertToRegistration"] = {"Status" : constants.statusDone}
        if os.path.isfile(os.path.join(folderPath, "result.json")):
            results = utilities.getResultObject(folderPath)
            if 'Height' in results:
                if results['Height'] != -1:
                    pipelineStatus["CalculateSize"] = {"Status" : constants.statusDone}
            if 'BackgroundRegistration' in results:
                if results['BackgroundRegistration']['Scale'] != -1:
                    pipelineStatus["BackgroundRegistration"] = {"Status" : constants.statusDone}
            if 'GrowthSinceLastSnapshot' in results:
                if results['GrowthSinceLastSnapshot'] != -1:
                    pipelineStatus["CalculateGrowth"] = {"Status" : constants.statusDone}
        return pipelineStatus

    def getPipelineStatusBackground(self,folderPath):
        pipelineStatus  = {
            "ImagesUploaded" : {"Status" : constants.statusNotDone},
            "PointCloudGenerated" : {"Status" : constants.statusNotDone},
            "ShapenetFormat" : {"Status" : constants.statusNotDone},
            "ConvertToRegistration" : {"Status" : constants.statusNotDone}
        }

        if os.path.isdir(os.path.join( folderPath, "images")) and len(os.listdir(os.path.join( folderPath, "images"))) != 0:
            pipelineStatus["ImagesUploaded"] = {"Status" : constants.statusDone}
        if os.path.isfile(os.path.join( folderPath, "odm_filterpoints", "point_cloud.ply")):
            pipelineStatus["PointCloudGenerated"] = {"Status" : constants.statusDone}
        if os.path.isfile(os.path.join( folderPath, "shapenet", "point_cloudSS1.txt")):
            pipelineStatus["ShapenetFormat"] = {"Status" : constants.statusDone}
        if os.path.isfile(os.path.join(folderPath, 'shapenet', "registrationFormat.txt")):
            pipelineStatus["ConvertToRegistration"] = {"Status" : constants.statusDone}
        return pipelineStatus

    def restoreState(self,dataPath):
        self.state = {}

        dirs = [name for name in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath,name))]
        for d in dirs:
            if d != "predictions":
                currentDirPath = os.path.join(dataPath,d)
                currentTimestamps = dirs = [name for name in os.listdir(currentDirPath) if os.path.isdir(os.path.join(currentDirPath,name))]
                self.state[d] = {}
                for t in currentTimestamps:
                    if t != "background":
                        self.state[d][t] = self.getPipelineStatus(os.path.join(currentDirPath,t))
                    else:
                        self.state[d][t] = self.getPipelineStatusBackground(os.path.join(currentDirPath,t))
        return self.state

    def saveState(self, dataPath):
        with open(os.path.join(dataPath, "state.json"), 'w') as f:
            json.dump(self.state, f)

    def getTestSetState(self, testSet):
        if testSet not in self.state:
            return None
        return self.state[testSet] 

    def getTimeStampState(self, testSet, timeStamp):
        if testSet not in self.state:
            return None
        if timeStamp not in self.state[testSet]:
            return None
        return self.state[testSet][timeStamp]

    def updateState(self, stateUpdate, result):
        dataSet = stateUpdate["data"]["testSet"]
        timeStamp = stateUpdate["data"]["timeStamp"]

        if dataSet not in self.state:
            self.state[dataSet] = {}
        if timeStamp not in self.state[dataSet]:
            if timeStamp != "background":
                self.state[dataSet][timeStamp] = {
                    "ImagesUploaded" : {"Status" : constants.statusNotDone},
                    "PointCloudGenerated" : {"Status" : constants.statusNotDone},
                    "ShapenetFormat" : {"Status" : constants.statusNotDone},
                    "BackgroundSegmented" : {"Status" : constants.statusNotDone},
                    "LeavesSegmented" : {"Status" : constants.statusNotDone},
                    "LeaveStemSplit" : {"Status" : constants.statusNotDone},
                    "CountLeaves" : {"Status" : constants.statusNotDone},
                    "BackgroundRegistration" : {"Status" : constants.statusNotDone},
                    "ConvertToRegistration" : {"Status" : constants.statusNotDone},
                    "CalculateSize" : {"Status" : constants.statusNotDone},
                    "CalculateGrowth" : {"Status" : constants.statusNotDone}
                }
            else:
                self.state[dataSet][timeStamp] = {
                    "ImagesUploaded" : {"Status" : constants.statusNotDone},
                    "PointCloudGenerated" : {"Status" : constants.statusNotDone},
                    "ShapenetFormat" : {"Status" : constants.statusNotDone},
                    "ConvertToRegistration" : {"Status" : constants.statusNotDone}
                }

        if stateUpdate["jobName"] == "SaveImages":
            self.state[dataSet][timeStamp]["ImagesUploaded"] = result
        if stateUpdate["jobName"] == "GeneratePointCloud":
            self.state[dataSet][timeStamp]["PointCloudGenerated"] = result
        if stateUpdate["jobName"] == "ConvertToShapenet":
            self.state[dataSet][timeStamp]["ShapenetFormat"] = result
        if stateUpdate["jobName"] == "SegmentLeaves":
            self.state[dataSet][timeStamp]["LeavesSegmented"] = result
        if stateUpdate["jobName"] == "SegmentBackground":
            self.state[dataSet][timeStamp]["BackgroundSegmented"] = result
        if stateUpdate["jobName"] == "LeaveStemSplit":
            self.state[dataSet][timeStamp]["LeaveStemSplit"] = result
        if stateUpdate["jobName"] == "CountLeaves":
            self.state[dataSet][timeStamp]["CountLeaves"] = result
        if stateUpdate["jobName"] == "BackgroundRegistration":
            self.state[dataSet][timeStamp]["BackgroundRegistration"] = result
        if stateUpdate["jobName"] == "ConvertToRegistration":
            self.state[dataSet][timeStamp]["ConvertToRegistration"] = result
        if stateUpdate["jobName"] == "CalculateSize":
            self.state[dataSet][timeStamp]["CalculateSize"] = result
        if stateUpdate["jobName"] == "CalculateGrowth":
            self.state[dataSet][timeStamp]["CalculateGrowth"] = result

        return self.state