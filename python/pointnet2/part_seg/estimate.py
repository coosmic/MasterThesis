import argparse
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
import part_dataset_all_normal

import debug

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_part_seg', help='Model name [default: pointnet2_part_seg]')
parser.add_argument('--model_path2', default='./pointnet2/part_seg/results/t3_2ClassesPartSeg_1024_256/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--model_path3', default='./pointnet2/part_seg/results/t4_3ClassesPartSeg/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--log_dir', default='./pointnet2/part_seg/log_eval', help='Log dir [default: log_eval]')
parser.add_argument('--num_point', type=int, default=16384, help='Point Number [default: 2048]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--pred_dir', default='./pointnet2/part_seg/pred_eval/', help='Directory where predictions should be saved to')
FLAGS = parser.parse_args()

#VOTE_NUM = 12
#
#EPOCH_CNT = 0
#
#BATCH_SIZE = FLAGS.batch_size
#NUM_POINT = FLAGS.num_point
#GPU_INDEX = FLAGS.gpu
#
#MODEL_PATH = FLAGS.model_path
#MODEL = importlib.import_module(FLAGS.model) # import network module
#MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
#LOG_DIR = FLAGS.log_dir
#if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
#os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
#os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
#LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
#LOG_FOUT.write(str(FLAGS)+'\n')
#NUM_CLASSES = 2
#
#PRED_DIR = FLAGS.pred_dir
#
## Shapenet official train/test split
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'shapenetcore_partanno_segmentation_benchmark_v0_normal')
#TEST_DATASET = part_dataset_all_normal.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, split='val')

BATCH_SIZE = 1
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu

#MODEL2C = None
#SESSION_2C = None
#OPS_2C = None
#DATALOADER_2C = None

class Pointnet2PartSegmentation:

    def __init__(self, model="pointnet2_part_seg", model_path="./pointnet2/part_seg/results/t3_2ClassesPartSeg_1024_256/model.ckpt", segmentationClasses = {'Plant': [0, 1] }):
        self.startPointnet2PartSegmentation(model, model_path, segmentationClasses)
        pass

    def startPointnet2PartSegmentation(self, model, model_path, segmentationClasses):
        print("pointnet2 starting")
        self.MODEL2C = importlib.import_module(model)
        with tf.Graph().as_default():
            with tf.device('/gpu:'+str(GPU_INDEX)):
                pointclouds_pl, labels_pl = self.MODEL2C.placeholder_inputs(BATCH_SIZE, NUM_POINT)
                is_training_pl = tf.placeholder(tf.bool, shape=())
                print("Is Training? ",is_training_pl)

                print("--- Get model and loss")
                pred, end_points = self.MODEL2C.get_model(pointclouds_pl, is_training_pl)
                loss = self.MODEL2C.get_loss(pred, labels_pl)
                saver = tf.train.Saver()

            # Create a session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            self.SESSION_2C = tf.Session(config=config)
            # Restore variables from disk.
            print(labels_pl)
            print(pointclouds_pl)
            saver.restore(self.SESSION_2C, model_path)
            self.OPS_2C = {'pointclouds_pl': pointclouds_pl,
                   'labels_pl': labels_pl,
                   'is_training_pl': is_training_pl,
                   'pred': pred,
                   'loss': loss} 
            self.DATALOADER_2C = part_dataset_all_normal.SingleFileDataset(root=DATA_PATH, npoints=NUM_POINT, segClasses=segmentationClasses)
        pass

    def get_batch(self, dataset, path):
        batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 6))
        batch_label = np.zeros((BATCH_SIZE, NUM_POINT), dtype=np.int32)
        ps,normal = dataset[path]
        batch_data[0,:,0:3] = ps
        batch_data[0,:,3:6] = normal
        return batch_data, batch_label

    def estimate(self, pathToPointCloud):
        print("CloudLeave estimation for ", pathToPointCloud)

        seg_classes = self.DATALOADER_2C.seg_classes
        seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        batch_data, batch_label = self.get_batch(self.DATALOADER_2C, pathToPointCloud)
        #print(batch_data)
        #print(batch_label)
        feed_dict = {self.OPS_2C['pointclouds_pl']: batch_data,
                     self.OPS_2C['labels_pl']: batch_label,
                     self.OPS_2C['is_training_pl']: False}
        #print(SESSION_2C)
        pred_val = np.zeros((BATCH_SIZE, NUM_POINT, 2))
        temp_loss_val, pred_val = self.SESSION_2C.run([self.OPS_2C['loss'], self.OPS_2C['pred']], feed_dict=feed_dict)

        cur_pred_val = pred_val[0:BATCH_SIZE]

        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((BATCH_SIZE, NUM_POINT)).astype(np.int32)
        for i in range(BATCH_SIZE):
            cat = seg_label_to_cat[batch_label[i,0]]
            logits = cur_pred_val_logits[i,:,:]
            cur_pred_val[i,:] = np.argmax(logits[:,seg_classes[cat]], 1) + seg_classes[cat][0]
        print("cur_pred_val.shape ",cur_pred_val.shape)
        segp = cur_pred_val[0,:]
        print("segp.shape", segp.shape)

        debug.showSegmentation(batch_data[0,:,0:3], segp)

        return pred_val[0]
