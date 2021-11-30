import os
import threading
import time
from threading import Thread, Lock
import atexit
import signal
import sys
from pointnet2.part_seg import estimate
import jobs
import utilities
import state
import traceback
import constants

from multiprocessing.managers import BaseManager

from flask import Flask, request
app = Flask(__name__)

class ServiceExit(Exception):
    """
    Custom exception which is used to trigger the clean exit
    of all running threads and the main program.
    """
    pass

class StateManager(BaseManager): pass

def Manager():
    m = StateManager()
    m.start()
    return m

StateManager.register('State', state.State)

global mutexJobQueue
mutexJobQueue = Lock()
global serverRunning
serverRunning = True
global jobqueue
jobqueue = []

global mutexImageQueue
mutexImageQueue = Lock()

global serverState
serverState = {}
global mutexServerState
mutexServerState = Lock()

global debug
debug = True

global mutexResult
mutexResult = Lock()

def jobProcessingThread(state):

    print("Job Processing Thread started")

    while serverRunning:
        if(len(jobqueue) != 0):
            try:
                mutexJobQueue.acquire()
                job = jobqueue[0]
                mutexJobQueue.release()

                if "FailedExecutionCount" in job:
                    failCount = job["FailedExecutionCount"]
                    if failCount > 5:
                        mutexJobQueue.acquire()
                        del jobqueue[0]
                        mutexJobQueue.release()
                        continue

                result = job["job"](job["data"])
                mutexServerState.acquire()
                state.updateState(job, result)
                mutexServerState.release()

                if("Results" in result):
                    mutexResult.acquire()
                    utilities.updateResultObject(os.path.join(os.path.abspath(os.getcwd()), "data", job["data"]["testSet"], job["data"]["timeStamp"]), result["Results"])
                    mutexResult.release()

                mutexJobQueue.acquire()
                del jobqueue[0]
                mutexJobQueue.release()
                
            except Exception as e:
                # If Server State is locked this has to be unlocked first to update server state
                if mutexServerState.locked():
                    mutexServerState.release()

                mutexServerState.acquire()
                state.updateState(job, {"Status": constants.statusFailed, "Reason": "Exception Occurred", "Details" : str(e)})
                mutexServerState.release()
                traceback.format_exc()

                if "FailedExecutionCount" not in job:
                    job["FailedExecutionCount"] = 0
                job["FailedExecutionCount"] = job["FailedExecutionCount"] + 1

                if mutexJobQueue.locked():
                    mutexJobQueue.release()
                if mutexResult.locked():
                    mutexResult.release()
                if debug:
                    raise e
        time.sleep(1.0)
    print("cleanup job processing queue")
    # If anything has to be cleaned up, here is the right place to do so. Right now we don't need any cleanup.
    print("jobProcessingThread stopped")

jobProcessingThreadInstance = threading.Thread(target=jobProcessingThread)

def atExitBackgroundThreads():
    print("Server shutdown gracefully")
    pass

def tearDownBackgroundThreads(sig, frame):
    print("background jobs stopping")
    global serverRunning
    serverRunning = False
    jobProcessingThreadInstance.join()
    print("background jobs stopped")
    sys.exit(0)

def startBackgroundThreads():
    print("background jobs starting")

    manager = Manager()
    state = manager.State()
    state.restoreState( os.path.join(os.path.abspath(os.getcwd()), "data") )
    global serverState
    serverState = state

    global jobProcessingThread
    jobProcessingThreadInstance = threading.Thread(target=jobProcessingThread, args=(serverState,))
    jobProcessingThreadInstance.start()

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

@app.route('/listing/<testSet>', methods=['GET'])
def listing(testSet):
    mutexServerState.acquire()

    state = serverState.getTestSetState(testSet)
    if not state is None:
        mutexServerState.release()
        return state

    mutexServerState.release()
    return("Not Found", 404)

@app.route('/result/<testSet>/<timeStamp>', methods=['GET'])
def results(testSet, timeStamp):
    path = os.path.join(os.path.abspath(os.getcwd()), "data", testSet, timeStamp)
    mutexResult.acquire()
    try:
        result = utilities.getResultObject(path)
        mutexResult.release()
        return(result, 200)
    except Exception as e:
        mutexResult.release()
        return("Not Found", 404)

@app.route('/detail/<testSet>/<timeStamp>', methods=['GET','POST', 'PUT'])
def data(testSet, timeStamp):
    if request.method == 'POST':
        mutexImageQueue.acquire()
        print(request.files.keys())
        if "images" not in request.files.keys():
            return ("Bad Request. You have to provide field images! ", 400)
        uploaded_files = request.files.getlist("images")
        print(uploaded_files)
        image_folder_path = os.path.join(os.path.abspath(os.getcwd()), "data", testSet, timeStamp, "images")
        
        if not os.path.isdir(image_folder_path):
            os.makedirs(image_folder_path)
        for file in uploaded_files:
            filePath = os.path.join(image_folder_path, file.filename)
            file.save(filePath)

        mutexImageQueue.release()
        utilities.createResultObject(os.path.join(os.path.abspath(os.getcwd()), "data", testSet, timeStamp))

        job2 = {"job" : jobs.jobGeneratePointCloud, "jobName" : "GeneratePointCloud", "data" : {"testSet" : testSet, "timeStamp" : timeStamp}}
        mutexJobQueue.acquire()
        jobqueue.append(job2)
        mutexJobQueue.release()
        return("OK")
    elif request.method == 'GET':
        print(testSet, timeStamp)
        mutexServerState.acquire()

        state = serverState.getTimeStampState(testSet, timeStamp)
        if not state is None:
            mutexServerState.release()
            return state

        mutexServerState.release()
        return("Not Found", 404)
    elif request.method == 'PUT':
        print(f"Updating {testSet}/{timeStamp}")
        if request.is_json:
            payloadR = request.get_json()
            jobList = payloadR["jobs"]
            for job in jobList:
                job = jobs.getJob(job['jobName'], testSet, timeStamp, job['jobParameter'])
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