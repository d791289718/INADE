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
from collections import OrderedDict
import glob
from loguru import logger
import cv2
import numpy as np
from data.GID_dataset import GIDDataset
from PIL import Image
from util.util import tensor2im, tensor2label
from util.metric import Evaluator
from util.visualizer import Visualizer
from util import html
# from tensorflow.keras.metrics import MeanIoU
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from models.networks.segmentor import UnetSegmentor



def test(opts):
    visualizer = Visualizer(opts)

    nc = opts.label_nc + 1 if opts.contain_dontcare_label else opts.label_nc
    # logger.add(os.path.join(opts.infer_dir, "infer.lg"))
    metric = Evaluator(opts.label_nc, os.path.join(os.path.join(opts.results_dir, opts.name), "infer.lg"), True)

    cps_all = glob.glob(os.path.join(os.path.join(opts.checkpoints_dir, opts.name), '*.pth'))
    cp_list = []
    cp_list.append("/home/wcd/repos/INADE/checkpoints/GID_v4_seg/latest_net_S.pth")
    cp_list += [data for data in cps_all if '.pth' in data] #  and data[-7:-4].isdigit() and 100<=int(data[-7:-4])<=499
    # cp_list.sort(reverse=True)
    

    val_data = GIDDataset()
    opts.phase = 'val'
    val_data.initialize(opt=opts)
    val_data = DataLoader(val_data, batch_size=opts.batchSize, shuffle=False, num_workers=8)

    test_data = GIDDataset()
    opts.phase = 'test'
    test_data.initialize(opt=opts)
    test_data = DataLoader(test_data, batch_size=opts.batchSize, shuffle=False, num_workers=8)

    classifier = UnetSegmentor(opts)

    best_val_miou = 0
    best_test_miou = 0
    for resume in cp_list:
        # load
        checkpoint = torch.load(resume)
        classifier.load_state_dict(checkpoint)

        classifier.cuda()
        classifier.eval()
        
        metric.reset()
        with torch.no_grad():
            for _, da, in enumerate(val_data):

                img, mask = da['image'], da['label']

                img = img.cuda()
                mask = mask.cuda()

                y_pred = classifier(img)
                y_pred = torch.log_softmax(y_pred, dim=1)
                _, y_pred = torch.max(y_pred, dim=1) # [bz, 256, 256]
                y_pred = y_pred.cpu().detach().numpy()
                mask = mask.cpu().detach().numpy()[:,0,:,:]

                metric.add_batch(mask, y_pred)
            mean_iou_val = metric.mIoU()
            metric.log(resume)

            # test
            if mean_iou_val > best_val_miou:
                web_dir = os.path.join(opts.results_dir, opts.name,
                       '%s_%s' % (opts.phase, resume))
                webpage = html.HTML(web_dir,
                                    'Experiment = %s, Phase = %s, Epoch = %s' %
                                    (opts.name, opts.phase, resume))


                best_val_miou = mean_iou_val
                metric.reset()
                with torch.no_grad():
                    for _, da, in enumerate(test_data):

                        img, mask = da['image'], da['label']

                        img = img.cuda()
                        mask = mask.cuda()
                        
                        y_pred = classifier(img)
                        y_pred = torch.log_softmax(y_pred, dim=1)
                        _, y_pred = torch.max(y_pred, dim=1)

                        for b in range(img.shape[0]):
                            visuals = OrderedDict([('image', img[b]), ('seg_gt', da['label'][b]),('seg_result', y_pred[b][None])])
                            visualizer.save_images(webpage, visuals, da['path'][b:b + 1])


                        y_pred = y_pred.cpu().detach().numpy()
                        mask = mask.cpu().detach().numpy()[:,0,:,:]

                        metric.add_batch(mask, y_pred)

                    best_test_miou = metric.mIoU()
                    metric.log(resume)
                    logger.info("Best Test IOU: {}, epoch: {}".format(best_test_miou, resume))
                    webpage.save()

if __name__ == '__main__':
    
    opts = TestOptions().parse()

    print("Opt", opts)

    path = os.path.join(opts.results_dir, opts.name)
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    test(opts)

