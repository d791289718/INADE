from collections import defaultdict
import random
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.seed_file = os.path.join(opt.aug_dir, 'aug_seed.txt')
        # self.seed_dict = {}

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        self.label_nc = self.opt.label_nc + (1 if self.opt.contain_dontcare_label else 0)
        self.embeddings = nn.Parameter(torch.Tensor(self.label_nc, opt.embedding_nc, 4)) # [16, 128, 4]
        self.init_embeddings()

        if opt.use_vae or 'inade' in opt.norm_mode:
            # In case of VAE, we will sample from random z vector
            # In case of INADE, a random sampled z vector is fed to the generator
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def pre_process_noise(self, noise, z):
        '''
        noise: [n,inst_nc,2,noise_nc], z_i [n,inst_nc,embedding_nc]
        z: [gamma_scales, gamma_biass, beta_scales, beta_biass], size=[n,inst_nc,embedding_nc]
        '''
        #! option4: 直接相乘 or 再次affine
        # s_noise = torch.unsqueeze(noise[:,:,0,:].mul(z[1])+z[0],2)
        # b_noise = torch.unsqueeze(noise[:,:,1,:].mul(z[3])+z[2],2)
        # return torch.cat([s_noise,b_noise],2)
        # 再次affine
        return torch.stack([noise[:,0,:,:].mul(z[0])+z[1], noise[:,1,:,:].mul(z[2])+z[3]], 1)

    def init_embeddings(self):
        nn.init.uniform_(self.embeddings[..., 0])
        nn.init.uniform_(self.embeddings[..., 2])
        nn.init.zeros_(self.embeddings[..., 1])
        nn.init.zeros_(self.embeddings[..., 3])

    def get_embedding(self, noise):
        # [16, 128, 2], [B, 16, 128] -> [B, 16, 128]
        # self.embdding [gamma_scale, gamma_bias, beta_scale, beta_bias]
        gamme_embedding = noise * self.embeddings[..., 0] + self.embeddings[..., 1]
        beta_embedding = noise * self.embeddings[..., 2] + self.embeddings[..., 3]
        return torch.stack([gamme_embedding, beta_embedding], 1)

    def forward(self, input, z=None, input_instances=None, noise=None, noise_ins=None, path=None): # noise_ins is the input, noise is the noise
        seg = input

        # Part 1. Process the input
        if self.opt.use_vae and 'spade' in self.opt.norm_mode:
            # SPADE - vae mode, z is the random noise 作为整个网络的输入
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        elif 'inade' in self.opt.norm_mode:
            # INADE feeds the random noise as the input of generator
            if noise_ins is None:
                noise_ins = torch.randn(input.size(0), self.opt.z_dim,
                                    dtype=torch.float32, device=input.get_device())

            x = self.fc(noise_ins)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        # Part 2. Process the noise for INADE if necessary
        if 'inade' in self.opt.norm_mode:
            if noise is None:
                noise = torch.randn([x.size()[0], self.label_nc, self.opt.embedding_nc], device=x.get_device()) # [B, label_nc, embedding_nc]
                embeddings = self.get_embedding(noise) # [B, label_nc, embedding_nc]
            if self.opt.use_vae:
                # z is the list of [gamma_scales, gamma_biass, beta_scales, beta_biass], [n,inst_nc,embedding_nc]
                embeddings = self.pre_process_noise(embeddings, z)
        else:
            embeddings = None

        #! option2: 要不要在进入每个层前做一个公用的MLP（要不要不同语义不同MLP）

        # if self.opt.phase == 'generate' and self.opt.record_noise:
        #     with open(self.seed_file, 'a') as f:
        #         for b in range(input.shape[0]):
        #             f.write(path[b])
        #             f.write('noise_0: ' + str(noise[0][b].tolist()) + '\n')
        #             f.write('noise_1: ' + str(noise[1][b].tolist()) + '\n')
        #             f.write('noise_ins: ' + str(noise_ins[b].tolist()) + '\n')
        #             f.write('\n')

        # Part 3. Forward the main branch
        x = self.head_0(x, seg, input_instances, embeddings)

        x = self.up(x)
        x = self.G_middle_0(x, seg, input_instances, embeddings)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg, input_instances, embeddings)

        x = self.up(x)
        x = self.up_0(x, seg, input_instances, embeddings)
        x = self.up(x)
        x = self.up_1(x, seg, input_instances, embeddings)
        x = self.up(x)
        x = self.up_2(x, seg, input_instances, embeddings)
        x = self.up(x)
        x = self.up_3(x, seg, input_instances, embeddings)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg, input_instances, embeddings)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)

        return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)
