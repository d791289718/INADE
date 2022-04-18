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
import sys
sys.path.append('../')
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import argparse
import gc
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import glob
import json
import cv2
import numpy as np
from data.dlrsd_dataset import DLRSDDataset
import scipy.misc
from PIL import Image
from util.util import tensor2im, tensor2label

def colorize_mask(mask, palette=None):
    palette =[ 255,  255,  255, # 0 background
        238,  229,  102,# 1 back_bumper
        0, 0, 0,# 2 bumper
        124,  99 , 34, # 3 car
        193 , 127,  15,# 4 car_lights
        248  ,213 , 42, # 5 door
        220  ,147 , 77, # 6 fender
        99 , 83  , 3, # 7 grilles
        116 , 116 , 138,  # 8 handles
        200  ,226 , 37, # 9 hoods
        225 , 184 , 161, # 10 licensePlate
        142 , 172  ,248, # 11 mirror
        153 , 112 , 146, # 12 roof
        38  ,112 , 254, # 13 running_boards
        229 , 30  ,141, # 14 tailLight
        52 , 83  ,84, # 15 tire
        194 , 87 , 125, # 16 trunk_lids
        225,  96  ,18,  # 17 wheelhub
        31 , 102 , 211, # 18 window
        104 , 131 , 101# 19 windshield
    ]
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask[0].astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return np.array(new_mask.convert('RGB'))

def test(opts):

    cps_all = glob.glob(os.path.join(opts.train_dir, '*.pth'))
    cp_list = [data for data in cps_all if '.pth' in data and 'BEST' not in data and data[-6].isdigit() and int(data[-6:-4]) > 40]

    test_data = DLRSDDataset()
    test_data.initialize(opt=opts)

    ids = range(opts.label_nc)
    
    # vis
    vis_data = DataLoader(test_data, batch_size=opts.batch_size, shuffle=False, num_workers=16)
    vis = []
    for j, da, in enumerate(vis_data):
        img, mask = da['image'], da['label']
        img = tensor2im(img[0])
        mask = tensor2label(mask[0], opts.label_nc+2)
        curr_vis = np.concatenate([img, mask], 0)
        if len(vis) < 50:
            vis.append(curr_vis)
    vis = np.concatenate(vis, 1)
    cv2.imwrite(os.path.join(opts.infer_dir, "testing_gt.jpg"), vis)

    test_data = DataLoader(test_data, batch_size=opts.batch_size, shuffle=False, num_workers=16)
    classifier = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=False,
                                                                     num_classes=opts.label_nc, aux_loss=None)
    classifier.val()
    cp_list.sort()
    best_val_miou = 0
    for resume in cp_list:
        checkpoint = torch.load(resume)
        classifier.load_state_dict(checkpoint['model_state_dict'])


        classifier.cuda()
        classifier.eval()

        unions = {}
        intersections = {}
        for target_num in ids:
            unions[target_num] = 0
            intersections[target_num] = 0

        with torch.no_grad():
            for _, da, in enumerate(test_data):

                img, mask = da['image'], da['label']

                img = img.cuda()
                mask = mask.cuda()

                y_pred = classifier(img)['out']
                y_pred = torch.log_softmax(y_pred, dim=1)
                _, y_pred = torch.max(y_pred, dim=1)
                y_pred = y_pred.cpu().detach().numpy()
                mask = mask.cpu().detach().numpy()

                curr_iou = []

                for target_num in ids:
                    y_pred_tmp = (y_pred == target_num).astype(int)
                    mask_tmp = (mask == target_num).astype(int)

                    intersection = (y_pred_tmp & mask_tmp).sum()
                    union = (y_pred_tmp | mask_tmp).sum()

                    unions[target_num] += union
                    intersections[target_num] += intersection

                    if not union == 0:
                        curr_iou.append(intersection / union)
            mean_ious = []

            for target_num in ids:
                mean_ious.append(intersections[target_num] / (1e-8 + unions[target_num]))
            mean_iou_val = np.array(mean_ious).mean()
            print("Checkpoints_{}: mIoU is {}".format(resume, mean_iou_val))

            # test
            if mean_iou_val > best_val_miou:
                best_val_miou = mean_iou_val
                unions = {}
                intersections = {}
                for target_num in ids:
                    unions[target_num] = 0
                    intersections[target_num] = 0

                with torch.no_grad():
                    testing_vis = []
                    for _, da, in enumerate(test_data):

                        img, mask = da['image'], da['label']

                        img = img.cuda()
                        mask = mask.cuda()

                        vis_img = tensor2im(img[0])
                        vis_mask = tensor2label(mask[0], opts.label_nc+2)
                        curr_vis = np.concatenate([vis_img, vis_mask], 0)
                        if len(testing_vis) < 50:
                            testing_vis.append(curr_vis)

                        y_pred = classifier(img)['out']
                        y_pred = torch.log_softmax(y_pred, dim=1)
                        _, y_pred = torch.max(y_pred, dim=1)
                        y_pred = y_pred.cpu().detach().numpy()
                        mask = mask.cpu().detach().numpy()

                        curr_iou = []

                        for target_num in ids:
                            y_pred_tmp = (y_pred == target_num).astype(int)
                            mask_tmp = (mask == target_num).astype(int)

                            intersection = (y_pred_tmp & mask_tmp).sum()
                            union = (y_pred_tmp | mask_tmp).sum()

                            unions[target_num] += union
                            intersections[target_num] += intersection

                            if not union == 0:
                                curr_iou.append(intersection / union)

                    testing_vis = np.concatenate(testing_vis, 1)
                    cv2.imwrite(os.path.join(opts.infer_dir, "testing.jpg"), vis)

                    test_mean_ious = []

                    for j, target_num in enumerate(ids):
                        iou = intersections[target_num] / (1e-8 + unions[target_num])
                        print("IOU for ", ids[j], iou)

                        test_mean_ious.append(iou)
                    best_test_miou = np.array(test_mean_ious).mean()
                    print("Best IOU ,", str(best_test_miou), "CP: ", resume)

    print("Validation mIOU:" ,best_val_miou)
    print("Testing mIOU:" , best_test_miou )

    result = {"Validation": best_val_miou, "Testing":best_test_miou}
    with open(os.path.join(opts.infer_dir, 're.json'), 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str)
    parser.add_argument('--resume', type=str,  default="")
    parser.add_argument('--batch_size', type=int,  default=50)
    parser.add_argument('--label_nc', type=int,  default=18)
    parser.add_argument('--epoch', type=int,  default=60)
    parser.add_argument('--train_dir', type=str,  default="deep_lab/checkpoints/dlrsd")
    parser.add_argument('--infer_dir', type=str,  default="deep_lab/results/dlrsd")
    parser.add_argument('--preprocess_mode', type=str,  default='resize_and_crop')
    parser.add_argument('--dataroot', type=str, default='./datasets/')
    parser.add_argument('--load_size', type=int, default=286)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--display_winsize=', type=int, default=256)
    parser.add_argument('--contain_dontcare_label', type=bool, default=False)
    parser.add_argument('--cache_filelist_read', type=bool, default=False)
    parser.add_argument('--cache_filelist_write', type=bool, default=False)
    parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize)
    parser.add_argument('--no_pairing_check', action='store_true')
    parser.add_argument('--no_flip', action='store_true')

    opts = parser.parse_args()
    opts.phase = 'test'
    opts.no_instance = True
    opts.isTrain = False
    
    print("Opt", opts)

    path = opts.infer_dir
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))


    test(opts)

