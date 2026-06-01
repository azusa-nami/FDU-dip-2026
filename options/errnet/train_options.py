from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)        
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')        
        self.parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training (Note: in train_errnet.py, we mannually tune the training protocol, but you can also use following setting by modifying the code in errnet_model.py)
        self.parser.add_argument('--nEpochs', '-n', type=int, default=60, help='# of epochs to run')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        self.parser.add_argument('--wd', type=float, default=1e-4, help='weight decay for optimizer')
        self.parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'], help='optimizer for generator and discriminator')
        self.parser.add_argument('--lr_policy', type=str, default='cosine', choices=['manual', 'cosine', 'step'], help='learning rate schedule')
        self.parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum learning rate for cosine schedule')
        self.parser.add_argument('--lr_step_size', type=int, default=20, help='epoch interval for step schedule')
        self.parser.add_argument('--lr_gamma', type=float, default=0.5, help='multiplicative factor for step schedule')
        self.parser.add_argument('--early_stop_patience', type=int, default=10, help='stop after this many eval rounds without improvement; <=0 disables')
        self.parser.add_argument('--early_stop_min_delta', type=float, default=0.0, help='minimum improvement needed for early stopping')
        self.parser.add_argument('--early_stop_metric', type=str, default='PSNR', choices=['PSNR', 'SSIM', 'NCC', 'LMSE'], help='metric averaged across validation datasets for early stopping')
        self.parser.add_argument('--eval_freq', type=int, default=5, help='evaluate every N epochs')
        self.parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='max gradient norm; <=0 disables clipping')
        self.parser.add_argument('--ema_decay', type=float, default=0.999, help='generator EMA decay; <=0 disables EMA')
        self.parser.add_argument('--no_ema_eval', action='store_true', help='do not use EMA weights for eval/test even if available')

        self.parser.add_argument('--low_sigma', type=float, default=2, help='min sigma in synthetic dataset')
        self.parser.add_argument('--high_sigma', type=float, default=5, help='max sigma in synthetic dataset')
        self.parser.add_argument('--low_gamma', type=float, default=1.3, help='max gamma in synthetic dataset')
        self.parser.add_argument('--high_gamma', type=float, default=1.3, help='max gamma in synthetic dataset')
        self.parser.add_argument('--synthesis', type=str, default='mixed', choices=['mixed', 'reflection2', 'advanced', 'legacy'], help='reflection synthesis used for synthetic training data')
        self.parser.add_argument('--synth_mix_ratio', type=str, default='0.8,0.2', help='mixing ratio for reflection2/advanced when synthesis=mixed')
        self.parser.add_argument('--fusion_mode', type=str, default='both', choices=['both', 'film', 'cross', 'none'], help='DINO fusion mode: both=FiLM+CrossAttn, film=FiLM only, cross=CrossAttn only, none=no DINO fusion (baseline)')
        self.parser.add_argument('--fusion_strided', action='store_true', help='if set, keep stride-2 downsampling even when DINO fusion is enabled')
        
        # data augmentation
        self.parser.add_argument('--batchSize', '-b', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=str, default='224,336,448', help='scale images to multiple size')
        self.parser.add_argument('--fineSize', type=str, default='224,224', help='then crop to this size')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')

        # for discriminator
        self.parser.add_argument('--which_model_D', type=str, default='disc_vgg', choices=['disc_vgg', 'disc_patch'])
        self.parser.add_argument('--gan_type', type=str, default='rasgan', help='gan/sgan : Vanilla GAN; rasgan : relativistic gan')
        
        # loss weight
        self.parser.add_argument('--unaligned_loss', type=str, default='vgg', help='unaligned loss: vgg|mse|ctx|ctx_vgg')
        self.parser.add_argument('--vgg_layer', type=int, default=31, help='VGG19 layer for the unaligned vgg loss')
        
        self.parser.add_argument('--lambda_gan', type=float, default=0.01, help='weight for gan loss')
        self.parser.add_argument('--lambda_vgg', type=float, default=0.05, help='weight for vgg loss')
        
        self.isTrain = True
