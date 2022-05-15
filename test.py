"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict
import numpy as np
import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
from util.util import save_image, tensor2im, get_file, save_label

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    generated = model(data_i, mode='inference')

    img_path = data_i['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b]), ('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

        if opt.phase == 'generate':
            _, name, _ = get_file(img_path[b])
            gen_name = name + '_' + opt.name + '_' + opt.which_epoch + '_' + opt.generate_stamp
            save_label(data_i['label'][b].cpu().long().numpy(), os.path.join(opt.aug_label_dir, gen_name+'.png'), opt.dataset_mode)
            save_label(data_i['label'][b].cpu().long().numpy(), os.path.join(opt.aug_instance_dir, gen_name+'.png'), opt.dataset_mode)
            save_image(tensor2im(generated[b]), os.path.join(opt.aug_img_dir, gen_name+'.png'))
            with open(os.path.join(opt.split_dir, opt.aug_txt), 'a') as f:
                f.write(gen_name+'\n')

webpage.save()
