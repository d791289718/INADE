"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class DLRSDDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        parser.set_defaults(dataroot='./datasets/')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=18)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        img_root = os.path.join(root, 'UCMerced_LandUse/Images')
        label_root = os.path.join(root, 'DLRSD/Images')
        instance_root = os.path.join(root, 'DLRSD/Instances')

        phase = 'val' if opt.phase == 'test' else 'train'
        # range_dic = {'train':range(95) , 'val':range(0,100)}
        # range_dic = {'train':range(95) , 'val':range(85,100)}
        range_dic = {'train': [], 'val': [], 'test': []}
        for i in range(10):
            range_dic['train'].extend(range(i*10+2, (i+1)*10-2))
            range_dic['val'].append(i*10)
            range_dic['test'].append(i*10+1)


        all_images = make_dataset(img_root, recursive=True, read_cache=False, write_cache=False)
        all_labels = make_dataset(label_root, recursive=True, read_cache=False, write_cache=False)
        all_instances = make_dataset(instance_root, recursive=True, read_cache=False, write_cache=False)
        all_images.sort()
        all_labels.sort()
        all_instances.sort()
        image_paths = []
        label_paths = []
        instance_paths = []
        for p in all_images:
            if int(os.path.splitext(p)[0][-2:]) in range_dic[phase]:
                image_paths.append(p)
        for p in all_labels:
            if int(os.path.splitext(p)[0][-2:]) in range_dic[phase]:
                label_paths.append(p)
        for p in all_instances:
            if int(os.path.splitext(p)[0][-2:]) in range_dic[phase]:
                instance_paths.append(p)

        return label_paths, image_paths, instance_paths

