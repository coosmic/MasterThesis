import os
import psutil
from .. import constants

import JobInterface

class JobGeneratePointCloud(JobInterface):

    def __init__(self):
        self.name = "JobGeneratePointCloud"
        self.state = {"Status": constants.statusNotDone}

    def execute(self, jobParameter):
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

        self.state = {"Status": constants.statusDone}
        return self.state
    
    def getStatus(self):
        return self.state