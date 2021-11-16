import argparse
import os
import numpy as np
import torch
import torch.utils.data
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from torch.utils.data import DataLoader
import open3d as o3d
import sys

from .learning3d.data_utils import UserData
from .learning3d.models import RPMNet, PPFNet
from .learning3d.models import DGCNN, DCP
from .learning3d.models import PointNet
from .learning3d.models import PointNetLK
from .learning3d.losses import FrobeniusNormLoss, RMSEFeaturesLoss

def display_open3d(template, source, transformed_source):
    template_ = o3d.geometry.PointCloud()
    source_ = o3d.geometry.PointCloud()
    transformed_source_ = o3d.geometry.PointCloud()
    template_.points = o3d.utility.Vector3dVector(template)
    source_.points = o3d.utility.Vector3dVector(source + np.array([0,0,0]))
    transformed_source_.points = o3d.utility.Vector3dVector(transformed_source)
    template_.paint_uniform_color([1, 0, 0])
    source_.paint_uniform_color([0, 1, 0])
    transformed_source_.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([template_, transformed_source_])

def get_transformations(igt):
    R_ba = igt[:, 0:3, 0:3]
    translation_ba = igt[:, 0:3, 3].unsqueeze(2)
    R_ab = R_ba.permute(0, 2, 1)
    translation_ab = -torch.bmm(R_ab, translation_ba)
    return R_ab, translation_ab, R_ba, translation_ba

def test_one_epoch(net, device, model, test_loader, withError=False, show=False, scale=1.0):
    model.eval()
    test_loss = 0.0
    pred  = 0.0
    count = 0
    rotation_errors, translation_errors, rmses = [], [], []
    test_error = 0.0

    for i, data in enumerate(tqdm(test_loader)):
        print(data)
        template, source, igt = data
        source = source * scale
        transformations = get_transformations(igt)
        transformations = [t.to(device) for t in transformations]
        R_ab, translation_ab, R_ba, translation_ba = transformations

        template = template.to(device)
        source = source.to(device)
        igt = igt.to(device)

        output = model(template, source)

        identity = torch.eye(3).cuda().unsqueeze(0).repeat(template.shape[0], 1, 1)
        loss_val = -1.0
        if net == 'PointNetLK':
            loss_val = FrobeniusNormLoss()(output['est_T'], igt) + RMSEFeaturesLoss()(output['r'])
        elif net == 'RPM':
            loss_val = FrobeniusNormLoss()(output['est_T'], igt)
        else:
            loss_val = torch.nn.functional.mse_loss(torch.matmul(output['est_R'].transpose(2, 1), R_ab), identity) + torch.nn.functional.mse_loss(output['est_t'], translation_ab[:,:,0])
            cycle_loss = torch.nn.functional.mse_loss(torch.matmul(output['est_R_'].transpose(2, 1), R_ba), identity) + torch.nn.functional.mse_loss(output['est_t_'], translation_ba[:,:,0])
            loss_val = loss_val + cycle_loss * 0.1

        if show:
            display_open3d(template.detach().cpu().numpy()[0,:,:3], source.detach().cpu().numpy()[0,:,:3], output['transformed_source'].detach().cpu().numpy()[0])

        if withError:
            error = 0
            cloud_registration_resulst = output['transformed_source']

            for i in range(template.shape[0]):
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(cloud_registration_resulst[i].cpu().detach().numpy())
                distances, indices = nbrs.kneighbors(template[i,:,0:3].cpu().detach().numpy())
                error = error + (np.sum(distances) / distances.shape[0])
            test_error = test_error + error

        test_loss += loss_val.item()
        count += 1
    test_loss = float(test_loss)/count
    if withError:
        test_error = float(test_error)/count
        return test_loss, error
    return test_loss

def test(args, model, test_loader, scale):
    test_loss, test_dist_sum = test_one_epoch(args.net, args.device, model, test_loader, True, args.show, scale)
    print('Validation Loss: %f & Validation Distance Sum: %f'%(test_loss, test_dist_sum))
    return test_dist_sum

def icp(src, target, threshold, scale, show=False):

    src = src * scale

    pcdSrc = o3d.geometry.PointCloud()
    pcdSrc.points = o3d.utility.Vector3dVector(src)
    
    pcdTarget = o3d.geometry.PointCloud()
    pcdTarget.points = o3d.utility.Vector3dVector(target)

    trans_init = np.identity(4)

    icp_res = o3d.pipelines.registration.registration_icp(
    pcdSrc, pcdTarget, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())

    tmp = pcdSrc.transform(icp_res.transformation)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(np.asarray(tmp.points)) #transformed src
    distances, indices = nbrs.kneighbors(np.asarray(pcdTarget.points)) # target
    error = (np.sum(distances) / distances.shape[0])

    if show:
        tmp.paint_uniform_color([1, 0, 0])
        pcdTarget.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([tmp, pcdTarget])

    return error, icp_res

def runWithDifferentScalesWithArgs(args, data, model = None):
    runWithDifferentScales(data, model, args.start_scale, args.end_scale, args.scale_step_width, args.show, args.icp, args.icp_threshold, args.net)

def runWithDifferentScales(data, model=None, start_scale = 1.0, end_scale = 3.0, scale_step_width=0.1, show=False, use_icp=True, icp_threshold = 0.2, net="PointNetLK" ):

    startScale = start_scale
    endScale = end_scale
    scaleStepWidth = scale_step_width

    assert startScale < endScale

    currentScale = startScale
    best_result = sys.float_info.max
    best_iteration = -1
    current_iteration = 0
    while currentScale <= endScale:
        if use_icp:
            result, _ = icp(data['source'][0], data['template'][0], icp_threshold, currentScale)
            if result < best_result:
                best_result = result
                best_iteration = current_iteration
        else:
            args = options()
            args.net = net
            args.scale = currentScale
            args.show = False
            result = test(args, model, data, currentScale)
            if result < best_result:
                best_result = result
                best_iteration = current_iteration

        #update scale
        currentScale = currentScale + scaleStepWidth
        current_iteration += 1
    print(f"best iteration {best_iteration} with score {best_result} and scale {startScale + (scaleStepWidth * best_iteration)}")

    if show:
        scale = startScale + (scaleStepWidth * best_iteration)
        if use_icp:
            err, result = icp(data['source'][0], data['template'][0], icp_threshold, scale, True)
        else:
            args.show = True
            test(args, model, data, scale)

    scale = startScale + (scaleStepWidth * best_iteration)
    if use_icp:
        err, result = icp(data['source'][0], data['template'][0], icp_threshold, scale, False)
        return result.transformation, scale
    else:
        #test(args, model, data, scale)  
        return None, scale #TODO Return Transformation from NN Registration
            

def loadRawDataWithoutArgs(srcPath, targetPath, numPoints):
    cloudSrc = np.loadtxt(srcPath).astype(np.float32)
    cloudTemplate = np.loadtxt(targetPath).astype(np.float32)

    print(cloudSrc.shape)
    print(cloudTemplate.shape)

    template = np.empty([1,numPoints,3])
    template[0] = cloudTemplate
    source = np.empty([1,numPoints,3])
    source[0] = cloudSrc
    transformation = np.ones([1,4,4])
    rawData = {'template' : template, 'source' : source, 'transformation' : transformation}
    return rawData

def loadDataLoader(srcPath, targetPath, numPoints):
    rawData = loadRawDataWithoutArgs(srcPath, targetPath, numPoints)
    dataset = UserData('registration', rawData)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=4)
    return loader

def loadRawData(args):
    cloudSrc = np.loadtxt(os.path.join(args.dataset_path,"SrcCloud.txt")).astype(np.float32)
    cloudTemplate = np.loadtxt(os.path.join(args.dataset_path,"TgtCloud.txt")).astype(np.float32)

    template = np.empty([1,args.num_points,3])
    template[0] = cloudTemplate
    source = np.empty([1,args.num_points,3])
    source[0] = cloudSrc
    transformation = np.ones([1,4,4])
    rawData = {'template' : template, 'source' : source, 'transformation' : transformation}
    return rawData

def loadModelWithArgs(args):
    return loadModel(args.net, args.device, "")

def loadModel(net, device="cuda:0", pathPrefix="Learning3D/"):
    model = None
    if net == 'PointNetLK':
        ptnet = PointNet(emb_dims=1024, use_bn=True)
        model = PointNetLK(feature_model=ptnet)
        model = model.to(device)
        assert os.path.isfile(pathPrefix+'checkpoints/exp_pnlk/models/best_model.t7')
        model.load_state_dict(torch.load(pathPrefix+'checkpoints/exp_pnlk/models/best_model.t7'), strict=False)
        model.to(device)
    elif net == 'DCP':
        dgcnn = DGCNN(emb_dims=512)
        model = DCP(feature_model=dgcnn, cycle=True)
        model = model.to(device)
        assert os.path.isfile(pathPrefix+'learning3d/pretrained/exp_dcp/models/best_model.t7')
        model.load_state_dict(torch.load(pathPrefix+'learning3d/pretrained/exp_dcp/models/best_model.t7'), strict=False)
        model.to(device)
    elif net == 'RPM':
        model = RPMNet(feature_model=PPFNet())
        model = model.to(device)
        assert os.path.isfile(pathPrefix+'learning3d/pretrained/exp_rpmnet/models/partial-trained.pth')
        model.load_state_dict(torch.load(pathPrefix+'learning3d/pretrained/exp_rpmnet/models/partial-trained.pth', map_location='cpu')['state_dict'])
        model.to(device)
    else:
        raise Exception(f"Unkown value for parameter net: {net}")
    return model

def options():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp_ipcrnet', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset_path', type=str, default='/home/solomon/Thesis/python/dcp/data/Custome/plant2t1',
                        metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
    parser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')
    parser.add_argument('--net', default='DCP', type=str, help='DCP, RPM, PointNetLK')
    parser.add_argument('--show', type=bool, default=False, help='Show Result')
    parser.add_argument('--scale', type=float, default=1.0, help='Scale should be applied on Source Cloud (Will not be used when estimate_scale is True)')
    parser.add_argument('--icp', type=bool, default=False, help='Register with ICP')
    parser.add_argument('--icp_threshold', type=float, default=0.2, help='ICP Threshold')

    parser.add_argument('--estimate_scale', type=bool, default=False, help='Estimate Scale of Source Cloud')
    parser.add_argument('--start_scale', type=float, default=1.0, help='Scale Estimation scale start')
    parser.add_argument('--end_scale', type=float, default=2.0, help='Scale Estimation scale end')
    parser.add_argument('--scale_step_width', type=float, default=0.1, help='Scale Estimation scale step width')

    # settings for input data
    parser.add_argument('--dataset_type', default='modelnet', choices=['modelnet', 'shapenet2'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--num_points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')

    # settings for PointNet
    parser.add_argument('--pointnet', default='tune', type=str, choices=['fixed', 'tune'],
                        help='train pointnet (default: tune)')
    parser.add_argument('--emb_dims', default=512, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')

    # settings for on training
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--pretrained', default='learning3d/pretrained/exp_dcp/models/best_model.t7', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')

    args = parser.parse_args()
    return args

def main():
    args = options()
    rawData = loadRawData(args)
    
    if args.icp:
        if args.estimate_scale:
            runWithDifferentScalesWithArgs(args, rawData)
        else:
            icp(rawData['source'][0], rawData['template'][0], args.icp_threshold, args.scale)
        return 0
    
    dataset = UserData('registration', rawData)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    model = loadModelWithArgs(args)
    if args.estimate_scale:
        runWithDifferentScalesWithArgs(args, loader, model)
    else:
        test(args, model, loader, args.scale)

    return 0

if __name__ == '__main__':
	main()