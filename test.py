import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import cv2
from net import DFMNet
from data import test_dataset
import time
from skimage.metrics import mean_squared_error, structural_similarity as ssim

def max_f_measure(gt, pred):
    pred_binary = (pred > 0.5).astype(np.float32)
    tp = np.sum((pred_binary == 1) & (gt == 1))
    fp = np.sum((pred_binary == 1) & (gt == 0))
    fn = np.sum((pred_binary == 0) & (gt == 1))

    if tp + fp == 0 or tp + fn == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0

    f_measure = 2 * (precision * recall) / (precision + recall)
    return f_measure

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path', type=str, default='./dataset/', help='test dataset path')
opt = parser.parse_args()

dataset_path = 'E:/guobiao/DFM-Net-Extension/data/RGBD_test_for_BTS-Net/'

# Set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print(f'USE GPU {opt.gpu_id}')

# Load the model
model = DFMNet()
model.load_state_dict(torch.load('./results/train/epoch_19.pth'))
model.cuda()
model.eval()

def save(res, gt, name, notation=None, sigmoid=True):
    res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze() if sigmoid else res.data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    save_file = os.path.join(save_path, name.replace('.png', f'_{notation}.png') if notation else name)

    cv2.imwrite(save_file, res * 255)

# test_datasets = ['NJU2K','NLPR','STERE', 'RGBD135', 'LFSD','SIP']

test_datasets = ['NLPR']
for dataset in test_datasets:
    with torch.no_grad():
        save_path = './results/benchmark/' + dataset
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = dataset_path + dataset + '/RGB/'
        gt_root = dataset_path + dataset + '/GT/'
        depth_root = dataset_path + dataset + '/depth/'
        test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)

        mse_list = []
        s_measure_list = []
        max_f_measure_list = []

        for i in range(test_loader.size):
            image, gt, depth, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()
            torch.cuda.synchronize()
            time_s = time.time()
            out = model(image, depth)
            torch.cuda.synchronize()
            time_e = time.time()
            t = time_e - time_s

            # Save the result
            save(out[1], gt, name)

            pre_s = F.interpolate(out[0], size=gt.shape, mode='bilinear', align_corners=False)
            pre = pre_s.sigmoid().data.cpu().numpy().squeeze()
            pred = (pre - pre.min()) / (pre.max() - pre.min() + 1e-8)
            # Calculate evaluation metrics

            # Calculate MSE
            mse = mean_squared_error(gt, pred)
            mse_list.append(mse)

            # Calculate S Measure
            s_measure = ssim(gt, pred, data_range=1)

            s_measure_list.append(s_measure)

            # Calculate Max F Measure
            f_measure = max_f_measure(gt, pred)
            max_f_measure_list.append(f_measure)

    print(f"Dataset: {dataset} testing completed.")
    print(f"MAE: {np.mean(mse_list):.4f}, S Measure (SSIM): {np.mean(s_measure_list):.4f}, Max F Measure: {np.mean(max_f_measure_list):.4f}")

print('Test Done!')
