from os import walk
import argparse
import numpy as np
import codecs, json 

parser = argparse.ArgumentParser()
parser.add_argument('--pathIn', default='/home/solomon/Thesis/python/data/plants/shapenet/OnlyCenter/', help='Path to Folder that contains Samples')
parser.add_argument('--pathOut', default='/home/solomon/Thesis/python/data/plants/shapenet/OnlyCenter/split/', help='Path to Folder that where Sets should be saved')
FLAGS = parser.parse_args()

filenames = next(walk(FLAGS.pathIn), (None, None, []))[2]

prefix = 'shape_data/13371337/'
filenames = [prefix + filename[0:-4] for filename in filenames]

numberOfSamples = len(filenames)

trainSetSize = int(numberOfSamples * 0.8)
testSetSize = int(numberOfSamples * 0.1)
valSetSize = int(numberOfSamples * 0.1)

print('Train Set Size: ', trainSetSize)
print('Test Set Size: ', testSetSize)
print('Validation Set Size: ', valSetSize)

trainSet = np.random.choice(filenames, trainSetSize, replace=False)
filenames = np.setdiff1d(filenames, trainSet)

testSet = np.random.choice(filenames, testSetSize, replace=False)
filenames = np.setdiff1d(filenames, testSet)

valSet = np.random.choice(filenames, valSetSize, replace=False)
filenames = np.setdiff1d(filenames, valSet)

trainSet = np.concatenate((trainSet, filenames))

assert(numberOfSamples == len(trainSet) + len(testSet) + len(valSet))

json.dump(trainSet.tolist(), codecs.open(FLAGS.pathOut+'shuffled_train_file_list.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=False)
json.dump(testSet.tolist(), codecs.open(FLAGS.pathOut+'shuffled_test_file_list.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=False)
json.dump(valSet.tolist(), codecs.open(FLAGS.pathOut+'shuffled_val_file_list.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=False)