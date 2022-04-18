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
import shutil
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import argparse
import gc
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision import transforms
from PIL import Image
# from utils.data_util import *
import json
from data.dlrsd_dataset import DLRSDDataset
import pickle

def main(opts):

    train_data = DLRSDDataset()
    train_data.initialize(opt=opts)
    

    train_data = DataLoader(train_data, batch_size=opts.batch_size, shuffle=True, num_workers=16)
    classifier = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=False,
                                                                     num_classes=opts.label_nc, aux_loss=None)
    if opts.resume != "":
        checkpoint = torch.load(opts.resume)
        classifier.load_state_dict(checkpoint['model_state_dict'])

    classifier.cuda()
    classifier.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    for epoch in range(opts.epoch):
        for i, da, in enumerate(train_data):
            classifier.train()

            optimizer.zero_grad()
            img, mask = da['image'], da['label']

            img = img.cuda()
            mask = mask.long().cuda()[:,0,:,:]

            y_pred = classifier(img)['out']
            loss = criterion(y_pred, mask)
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                print(epoch, 'epoch', 'iteration', i, 'loss', loss.item())

        model_path = os.path.join(opts.train_dir, 'deeplab_' + str(epoch) + '.pth')
        torch.save({'model_state_dict': classifier.state_dict()}, model_path)
        print('Save to:', model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str)
    parser.add_argument('--resume', type=str,  default="")
    parser.add_argument('--batch_size', type=int,  default=42)
    parser.add_argument('--label_nc', type=int,  default=18)
    parser.add_argument('--epoch', type=int,  default=60)
    parser.add_argument('--train_dir', type=str,  default="deep_lab/checkpoints/dlrsd")
    # parser.add_argument('--infer_dir', type=str,  default="deep_lab/results/dlrsd")
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
    opts.phase = 'train'
    opts.no_instance = True
    opts.isTrain = True
    
    print("Opt", opts)

    path = opts.train_dir
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    main(opts)


