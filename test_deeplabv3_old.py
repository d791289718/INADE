"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from logging import info
from re import M
import sys
sys.path.append('../')
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import argparse
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import glob
from loguru import logger
import cv2
import numpy as np
from data.GID_dataset import GIDDataset
from PIL import Image
from util.util import tensor2im, tensor2label
from util.metric import Evaluator
# from tensorflow.keras.metrics import MeanIoU
from options.test_options import TestOptions



def test(opts):
    nc = opts.label_nc + 1 if opts.contain_dontcare_label else opts.label_nc
    logger.add(os.path.join(opts.infer_dir, "infer.lg"))
    metric = Evaluator(opts.label_nc, os.path.join(opts.infer_dir, "infer.lg"), True)

    cps_all = glob.glob(os.path.join(opts.checkpoints_dir, '*.pth'))
    cp_list = [data for data in cps_all if '.pth' in data and 'BEST' not in data and data[-7:-4].isdigit() and 100<=int(data[-7:-4])<=145] #  and data[-6].isdigit() and int(data[-6:-4]) > 40 and 
    cp_list.sort(reverse=True)

    val_data = GIDDataset()
    opts.phase = 'val'
    val_data.initialize(opt=opts)
    val_data = DataLoader(val_data, batch_size=opts.batchSize, shuffle=False, num_workers=16)

    test_data = GIDDataset()
    opts.phase = 'test'
    test_data.initialize(opt=opts)
    test_data = DataLoader(test_data, batch_size=opts.batchSize, shuffle=False, num_workers=16)

    ids = range(opts.label_nc)
    
    # # vis
    # vis = []
    # for j, da, in enumerate(test_data):
    #     img, mask = da['image'], da['label']

    #     if len(vis) < 10:
    #         img = tensor2im(img[0])
    #         mask = tensor2label(mask[0], nc)
    #         curr_vis = np.concatenate([img, mask], 0)
    #         vis.append(curr_vis)
    # vis = np.concatenate(vis, 1)
    # cv2.imwrite(os.path.join(opts.infer_dir, "testing_gt.jpg"), vis)

    classifier = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=False,
                                                                     num_classes=nc, aux_loss=None)
    best_val_miou = 0
    best_test_miou = 0
    for resume in cp_list:
        checkpoint = torch.load(resume)
        classifier.load_state_dict(checkpoint['model_state_dict'])


        classifier.cuda()
        classifier.eval()

        # unions = {}
        # intersections = {}
        # for target_num in ids:
        #     unions[target_num] = 0
        #     intersections[target_num] = 0
        # curr_ious = []

        metric.reset()
        with torch.no_grad():
            for _, da, in enumerate(val_data):

                img, mask = da['image'], da['label']

                img = img.cuda()
                mask = mask.cuda()

                y_pred = classifier(img)['out']
                y_pred = torch.log_softmax(y_pred, dim=1)
                _, y_pred = torch.max(y_pred, dim=1) # [bz, 256, 256]
                y_pred = y_pred.cpu().detach().numpy()
                mask = mask.cpu().detach().numpy()[:,0,:,:]

                metric.add_batch(mask, y_pred)
            #     for target_num in ids:
            #         y_pred_tmp = (y_pred == target_num).astype(int)
            #         mask_tmp = (mask == target_num).astype(int)

            #         intersection = (y_pred_tmp & mask_tmp).sum()
            #         union = (y_pred_tmp | mask_tmp).sum()

            #         if union > 0: unions[target_num] += union
            #         if intersection > 0: intersections[target_num] += intersection

            #         if not union == 0:
            #             curr_ious.append(intersection / union)
            #     print(curr_ious)
            # mean_ious = []
            # for j, target_num in enumerate(ids):
            #     iou = intersections[target_num] / (1e-8 + unions[target_num])
            #     if iou > 0: mean_ious.append(iou)
            #     logger.info("Val IOU for {}: {}".format(ids[j], iou))
            # mean_iou_val = np.array(mean_ious).mean()
            # logger.info("Checkpoints_{}: mIoU is {}".format(resume, mean_iou_val))
            mean_iou_val = metric.mIoU()
            metric.log(resume)

            # test
            if mean_iou_val > best_val_miou:
                best_val_miou = mean_iou_val
                # unions = {}
                # intersections = {}
                # for target_num in ids:
                #     unions[target_num] = 0
                #     intersections[target_num] = 0
                # curr_ious = []

                metric.reset()
                with torch.no_grad():
                    testing_vis = []
                    for _, da, in enumerate(test_data):

                        img, mask = da['image'], da['label']

                        img = img.cuda()
                        mask = mask.cuda()
                        
                        y_pred = classifier(img)['out']
                        y_pred = torch.log_softmax(y_pred, dim=1)
                        _, y_pred = torch.max(y_pred, dim=1)

                        # if len(testing_vis) < 10:
                        #     vis_img = tensor2im(img[0])
                        #     vis_mask = tensor2label(torch.unsqueeze(y_pred[0], 0), nc)
                        #     curr_vis = np.concatenate([vis_img, vis_mask], 0)
                        #     testing_vis.append(curr_vis)
                        # testing_vis = np.concatenate(testing_vis, 1)
                        # cv2.imwrite(os.path.join(opts.infer_dir, "testing.jpg"), testing_vis)

                        y_pred = y_pred.cpu().detach().numpy()
                        mask = mask.cpu().detach().numpy()[:,0,:,:]


                        metric.add_batch(mask, y_pred)

                        # for target_num in ids:
                        #     y_pred_tmp = (y_pred == target_num).astype(int)
                        #     mask_tmp = (mask == target_num).astype(int)

                        #     intersection = (y_pred_tmp & mask_tmp).sum()
                        #     union = (y_pred_tmp | mask_tmp).sum()

                        #     if union > 0 :unions[target_num] += union
                        #     if intersection > 0:intersections[target_num] += intersection

                        #     if not union == 0:
                        #         curr_ious.append(intersection / union)

                    # test_mean_ious = []
                    # for j, target_num in enumerate(ids):
                    #     iou = intersections[target_num] / (1e-8 + unions[target_num])
                    #     logger.info("Test IOU for {}: {}".format(ids[j], iou))
                    #     if iou > 0: test_mean_ious.append(iou)
                    # best_test_miou = np.array(test_mean_ious).mean()
                    best_test_miou = metric.mIoU()
                    metric.log(resume)
                    logger.info("Best Test IOU: {}, epoch: {}".format(best_test_miou, resume))

    # logger.info("Validation mIOU:".format(best_val_miou))
    # logger.info("Testing mIOU:".format(best_test_miou))



if __name__ == '__main__':
    
    opts = TestOptions().parse()
    opts.infer_dir = os.path.join(opts.deeplab_infer_dir, opts.name)
    opts.checkpoints_dir = os.path.join(opts.deeplab_train_dir, opts.name)

    print("Opt", opts)

    path = opts.infer_dir
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    test(opts)

