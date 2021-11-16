from os import walk
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--pathIn', default='/home/solomon/Thesis/MasterThesis/python/pointnet2/part_seg/results/training/t8_2Classes_PartSeg_NoNorm_RandRot/log_train.txt', help='Path to File contains logs')
parser.add_argument('--plotName', default='T8_2C_PS_NN_RR', help='Name for the plot that should be generated')

FLAGS = parser.parse_args()

with open(FLAGS.pathIn, "r") as f:
    data = {}
    index = 0
    lines = f.readlines()
    for line in lines:
        if line.startswith("---- EPOCH"):
            
            data[index] = {
                "meanLoss" : -1,
                "accuracy" : -1,
                "mIoU" : -1 
            }
            index = index+1
        if line.startswith("eval mean loss"):
            data[len(data)-1]["meanLoss"] = float(line.replace("eval mean loss: ", ""))
        if line.startswith("eval accuracy: "):
            data[len(data)-1]["accuracy"] = float(line.replace("eval accuracy: ", ""))
        if line.startswith("eval mean mIoU: "):
            data[len(data)-1]["mIoU"] = float(line.replace("eval mean mIoU: ", ""))

    keys = data.keys()
    meanLoss = [ data[key]["meanLoss"] for key in data]
    accuracy = [ data[key]["accuracy"] for key in data]
    mIoU = [ data[key]["mIoU"] for key in data]

    # Create plots with pre-defined labels.
    fig, ax = plt.subplots()
    ax.plot(keys, meanLoss, 'r-', label='Loss')
    ax.plot(keys, accuracy, 'b-', label='Accuracy')
    ax.plot(keys, mIoU, 'g-', label='mIoU')

    legend = ax.legend(loc='best', shadow=False, fontsize='large')

    plt.title(FLAGS.plotName)
    plt.show()

