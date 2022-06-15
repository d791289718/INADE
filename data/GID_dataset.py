"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from email.policy import default
import os
from re import L
from data.pix2pix_dataset import Pix2pixDataset
from data.base_dataset import BaseDataset, get_params, get_transform
from torchvision.transforms.functional import InterpolationMode
from data.image_folder import make_dataset
from PIL import Image
import shutil


class GIDDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 286 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=15)
        parser.set_defaults(contain_dontcare_label=True)
        # parser.set_defaults(no_instance=True)

        parser.set_defaults(dataroot='./datasets/GID')
        parser.add_argument('--label_dir', type=str,  default='./datasets/GID/labels')
        parser.add_argument('--image_dir', type=str, default='./datasets/GID/images')
        parser.add_argument('--instance_dir', type=str, default='./datasets/GID/labels')
        parser.add_argument('--split_dir', type=str, default='./datasets/GID/split_txt_v3')
        parser.add_argument('--aug_dir', type=str, default='./datasets/GID/aug')
        parser.add_argument('--aug_img_dir', type=str, default='./datasets/GID/aug/images')
        parser.add_argument('--aug_label_dir', type=str, default='./datasets/GID/aug/labels')
        parser.add_argument('--aug_instance_dir', type=str, default='./datasets/GID/aug/labels')
        parser.add_argument('--generate_input_dir', type=str, default='./datasets/GID/labels')
        parser.add_argument('--aug_txt', type=str, default='aug.txt')
        return parser

    def get_paths(self, opt):
        # all_label_dir = opt.label_dir
        # all_label_paths = make_dataset(all_label_dir, recursive=False, read_cache=True)

        # all_image_dir = opt.image_dir
        # all_image_paths = make_dataset(all_image_dir, recursive=False, read_cache=True)

        # if len(opt.instance_dir) > 0:
        #     all_instance_dir = opt.instance_dir
        #     all_instance_paths = make_dataset(all_instance_dir, recursive=False, read_cache=True)
        # else:
        #     all_instance_paths = []

        image_paths, label_paths, instance_paths = [], [], []
        with open(os.path.join(opt.split_dir, opt.phase + '.txt'), 'r') as f:
            for name in f:
                name = name[:-1]
                if opt.phase != 'generate':
                    label_paths.append(os.path.join(opt.label_dir, name+'.png'))
                    if not opt.no_instance and len(opt.instance_dir) > 0:
                        instance_paths.append(os.path.join(opt.instance_dir, name+'.png'))
                    image_paths.append(os.path.join(opt.image_dir, name+'.png'))
                else:
                    label_paths.append(os.path.join(opt.generate_input_dir, name+'.png'))
                    if not opt.no_instance and len(opt.generate_input_dir) > 0:
                        instance_paths.append(os.path.join(opt.generate_input_dir, name+'.png'))
                    if opt.use_vae:
                        specific_style = False
                        if not specific_style:image_paths.append(os.path.join(opt.image_dir, name+'.png'))
        
        if opt.phase == 'generate' and opt.use_vae and specific_style:
            image_paths.append(os.path.join(opt.image_dir, 'GF2_PMS1__L1A0001118839-MSS1_4352_4864.png'))
            image_paths.append(os.path.join(opt.image_dir, 'GF2_PMS1__L1A0001118839-MSS1_4352_4864.png'))
            image_paths.append(os.path.join(opt.image_dir, 'GF2_PMS1__L1A0001118839-MSS1_4352_4864.png'))
            image_paths.append(os.path.join(opt.image_dir, 'GF2_PMS1__L1A0001118839-MSS1_4352_4864.png'))

        if opt.add_aug:
            with open(os.path.join(opt.split_dir, opt.aug_txt), 'r') as f:
                for name in f:
                    name = name[:-1]
                    label_paths.append(os.path.join(opt.aug_label_dir, name+'.png'))
                    if not opt.no_instance and len(opt.aug_instance_dir) > 0:
                        instance_paths.append(os.path.join(opt.aug_instance_dir, name+'.png'))
                    if opt.phase != 'generate':
                        image_paths.append(os.path.join(opt.aug_img_dir, name+'.png'))
        if opt.phase != 'generate' or opt.phase == 'generate' and opt.use_vae == True:
            assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"
        if opt.phase != 'generate':
            txt_root = opt.checkpoints_dir if opt.phase == "train" else opt.results_dir
            os.makedirs(os.path.join(txt_root, opt.name), exist_ok=True)
            file_name = os.path.join(os.path.join(txt_root, opt.name), 'dataset.txt')
            shutil.copyfile(os.path.join(opt.split_dir, 'readme.txt'), file_name)
            with open(file_name, 'a') as opt_file:
                if opt.add_aug: opt_file.write("\n add_gud: {}".format(opt.aug_txt))
                opt_file.write("\n[exactly in {}] dataset size: ".format(opt.phase))
                opt_file.write("label: {}, image: {}, instance: {}".format(len(label_paths), len(image_paths), len(instance_paths)))

        return label_paths, image_paths, instance_paths # paths

    def label_grb2gray(self, label):
        palette = [[200, 0, 0], # industrial land
                    [250, 0, 150], # urban residential
                    [200, 150, 150], # rural residential
                    [250, 150, 150], # traffic land
                    [0, 200, 0], # paddy field
                    [150, 250, 0], # irrigated land
                    [150, 200, 150], # dry cropland
                    [200, 0, 200], # garden plot
                    [150, 0, 250], # arbor woodland
                    [150, 150, 250], # shrub land
                    [250, 200, 0], # natural grassland
                    [200, 200, 0], # artificial grassland
                    [0, 0, 200], # river
                    [0, 150, 200], # lake
                    [0, 200, 250], # pond
                    [0, 0, 0]]
        
        color2index = {
            (200, 0, 0): 0, # industrial land
            (250, 0, 150): 1, # urban residential
            (200, 150, 150): 2, # rural residential
            (250, 150, 150): 3, # traffic land
            (0, 200, 0): 4, # paddy field
            (150, 250, 0): 5, # irrigated land
            (150, 200, 150): 6, # dry cropland
            (200, 0, 200): 7, # garden plot
            (150, 0, 250): 8, # arbor woodland
            (150, 150, 250): 9, # shrub land
            (250, 200, 0): 10, # natural grassland
            (200, 200, 0): 11, # artificial grassland
            (0, 0, 200): 12, # river
            (0, 150, 200): 13, # lake
            (0, 200, 250): 14, # pond
            (0, 0, 0): 15
        }
        return palette

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=InterpolationMode.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor = label_tensor.long()
        # label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # input image (real images)
        if self.opt.phase == 'generate' and not self.opt.use_vae:
            image_tensor = 0
            image_path = label_path
        else:
            image_path = self.image_paths[index]
            assert self.paths_match(label_path, image_path), \
                "The label_path %s and image_path %s don't match." % \
                (label_path, image_path)
            image = Image.open(image_path)
            image = image.convert('RGB')

            transform_image = get_transform(self.opt, params)
            image_tensor = transform_image(image)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode in ('L', 'P'):
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                print('something wrong')
                instance_tensor = transform_label(instance)

        input_dict = {'label': label_tensor, # [1, 256, 256]
                      'instance': instance_tensor, # [1, 256, 256]
                      'image': image_tensor, # [3, 256, 256]
                      'path': image_path,
                      }

        return input_dict # origin data