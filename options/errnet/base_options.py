from options.base_option import BaseOptions as Base
from util import util
from util.distributed import init_distributed, is_main_process
import os
import torch
import numpy as np
import random

class BaseOptions(Base):
    def initialize(self):
        Base.initialize(self)
        # experiment specifics
        self.parser.add_argument('--inet', type=str, default='errnet', help='chooses which architecture to use for inet.')
        self.parser.add_argument('--icnn_path', type=str, default=None, help='icnn checkpoint to use.')
        self.parser.add_argument('--init_type', type=str, default='edsr', help='network initialization [normal|xavier|kaiming|orthogonal|uniform]')
        # for network
        self.parser.add_argument('--hyper', action='store_true', help='if true, fuse frozen DINOv3 features inside the CNN backbone')
        self.parser.add_argument('--feature_model_path', type=str, default='/oldhome/zengyuqi/model/dinov3', help='local DINOv3 model directory')
        self.parser.add_argument('--feature_layers', type=str, default='6,12,18,24', help='DINOv3 hidden-state layers for perceptual losses')
        self.parser.add_argument('--hyper_layer', type=int, default=24, help='DINOv3 hidden-state layer fused inside the backbone')
        self.parser.add_argument('--feature_scale', type=float, default=0.1, help='scale applied to normalized DINOv3 hyper features')
        self.parser.add_argument('--no_feature_norm', action='store_true', help='disable normalization of DINOv3 hyper features before fusion')
        self.parser.add_argument('--fusion_type', type=str, default='film_cross', choices=['film_cross', 'film', 'cross_attn', 'none'], help='DINOv3 feature fusion type inside the backbone')
        self.parser.add_argument('--no_full_res', action='store_true', help='disable full-resolution processing (use original downsampled path)')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        init_distributed(self.opt)

        seed = self.opt.seed + self.opt.rank
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        args = vars(self.opt)

        if is_main_process(self.opt):
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

        # save to the disk
        self.opt.name = self.opt.name or '_'.join([self.opt.model])
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        if is_main_process(self.opt):
            util.mkdirs(expr_dir)
            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

        if self.opt.debug:
            self.opt.display_freq = 20
            self.opt.print_freq = 20
            self.opt.nEpochs = 40
            self.opt.max_dataset_size = 100
            self.opt.no_log = False
            self.opt.nThreads = 0
            self.opt.decay_iter = 0
            self.opt.serial_batches = True
            self.opt.no_flip = True
        
        return self.opt
