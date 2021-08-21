import os
import sys

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils/pointnet2'))
import tf_util
from pointnet_util import pointnet_fp_module, pointnet_sa_module
NUM_LABEL=2


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """ Part segmentation PointNet, input is BxNx6 (XYZ NormalX NormalY NormalZ), output B x num_classes """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])
    l0_points = tf.slice(point_cloud, [0, 0, 3], [-1, -1, 3])

    # Set Abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(
        l0_xyz, l0_points, npoint=2048, radius=0.1, nsample=32, mlp=[32, 32, 64],
        mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(
        l1_xyz, l1_points, npoint=256, radius=0.3, nsample=32, mlp=[64, 64, 128],
        mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(
        l2_xyz, l2_points, npoint=16, radius=0.8, nsample=32, mlp=[128, 128, 256],
        mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # Feature Propagation layers
    l2_points = pointnet_fp_module(
        l2_xyz, l3_xyz, l2_points, l3_points, [256, 256],
        is_training, bn_decay, scope='fa_layer2')
    l1_points = pointnet_fp_module(
        l1_xyz, l2_xyz, l1_points, l2_points, [256, 128],
        is_training, bn_decay, scope='fa_layer3')
    l0_points = pointnet_fp_module(
        l0_xyz, l1_xyz, l0_points, l1_points, [128, 128, 128],
        is_training, bn_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True,
                         is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, NUM_LABEL, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net, end_points


def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 2048, 6))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)
