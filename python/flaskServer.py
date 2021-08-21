import os
import threading
import time
from threading import Thread, Lock
import atexit
import signal
import sys
from pointnet2.part_seg import estimate
import jobs

from flask import Flask, request
app = Flask(__name__)

class ServiceExit(Exception):
    """
    Custom exception which is used to trigger the clean exit
    of all running threads and the main program.
    """
    pass

mutexJobQueue = Lock()
serverRunning = True
jobqueue = []

def jobProcessingThread():
    while serverRunning:
        mutexJobQueue.acquire()
        #print("thread waked up")
        #print(len(jobqueue))
        if(len(jobqueue) != 0):
            #print("processing job")
            try:
                jobqueue[0]["job"](jobqueue[0]["data"])
                del jobqueue[0]
            except Exception as e:
                print(e)
                mutexJobQueue.release()
                raise e
        mutexJobQueue.release()
        time.sleep(1.0)
    print("cleanup job processing queue")
    #TODO
    print("jobProcessingThread stopped")

jobProcessingThread = threading.Thread(target=jobProcessingThread)

#x = threading.Thread(target=jobThread)
#x.start()

def atExitBackgroundThreads():
    print("Server shutdown gracefully")
    pass

def tearDownBackgroundThreads(sig, frame):
    print("background jobs stopping")
    global serverRunning
    serverRunning = False
    jobProcessingThread.join()
    print("background jobs stopped")
    #raise ServiceExit
    sys.exit(0)

def startBackgroundThreads():
    print("background jobs starting")
    
    jobProcessingThread.start()

    jobs.startPartSegPipelines()

    atexit.register(atExitBackgroundThreads)
    signal.signal(signal.SIGINT, tearDownBackgroundThreads)
    print("background jobs started")

@app.route('/')
def index():
  return 'Index Page'

@app.route('/test')
def test():
    print("test")
    mutexJobQueue.acquire()
    job = {"job" : jobs.jobLeavesSegmentation, "data" : {"pathToPointCloud" : "/home/solomon/Thesis/python/data/plants/shapenet/withoutBackground/Avocado5LabeledShapenetSS1.txt"}}
    jobqueue.append(job)
    mutexJobQueue.release()
    return("OK")

@app.route('/data/<testSet>/<timeStamp>', methods=['GET','POST', 'PUT'])
def data(testSet, timeStamp):
    if request.method == 'POST':
        print(request.files.keys())
        uploaded_files = request.files.getlist("images")
        image_folder_path = os.path.join(os.path.abspath(os.getcwd()), "data", testSet, timeStamp, "images")
        base_folder_path = os.path.join(os.path.abspath(os.getcwd()), "data", )
        if not os.path.isdir(image_folder_path):
            os.makedirs(image_folder_path)
        for file in uploaded_files:
            filePath = os.path.join(image_folder_path, file.filename)
            #print(filePath)
            file.save(filePath)
        
        job = {"job" : jobs.jobGeneratePointCloud, "data" : {"baseFolderPath" : base_folder_path, "testSet" : testSet, "timeStamp" : timeStamp}}
        mutexJobQueue.acquire()
        jobqueue.append(job)
        mutexJobQueue.release()
        #makePointCloudCommand = f"docker run -ti --rm -v {base_folder_path}/{testSet}:/{testSet} --gpus all opendronemap/odm:gpu --rerun-all -e  odm_filterpoints --project-path /{testSet} {timeStamp}"
        #print(makePointCloudCommand)
        #os.system(makePointCloudCommand)
        return("OK")
    elif request.method == 'GET':
        #serve login page
        print(testSet, timeStamp)
        return("OK")
    elif request.method == 'PUT':
        print(f"Updating {testSet}/{timeStamp}")
        if request.is_json:
            payloadR = request.get_json()
            jobList = payloadR["jobs"]
            for job in jobList:
                print(f"Job with name {job['jobName']} will be started")
                job = {"job" : jobs.genericJob, "data" : {"jobName": job['jobName'], "jobParameter": job['jobParameter'],"testSet" : testSet, "timeStamp" : timeStamp}}
                mutexJobQueue.acquire()
                jobqueue.append(job)
                mutexJobQueue.release()
            return("OK")
        else:
            return("Please provide json data as payload and use Content-Type application/json!", 400)


if __name__ == "__main__":
    try:
        startBackgroundThreads()
        app.run(port=5000, host="localhost")
    except ServiceExit:
        print("Shutdown Signal")
        sys.exit()