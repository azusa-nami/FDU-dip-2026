import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

import os
import numpy as np
import itertools
from contextlib import contextmanager
from collections import OrderedDict

import util.util as util
import util.index as index
import models.networks as networks
import models.losses as losses
from models import arch

from .base_model import BaseModel
from PIL import Image
from os.path import join


def _unwrap_ddp(module):
    return module.module if isinstance(module, DistributedDataParallel) else module


def _clean_state_dict(state_dict):
    return {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _torch_load_compat(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _build_optimizer(opt, params):
    optimizer_cls = torch.optim.AdamW if opt.optimizer == 'adamw' else torch.optim.Adam
    return optimizer_cls(params, lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.wd)


def _copy_state_dict(state_dict, device=None):
    copied = {}
    for key, value in state_dict.items():
        tensor = value.detach().clone()
        if device is not None:
            tensor = tensor.to(device)
        copied[key] = tensor
    return copied


def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = image_numpy.astype(imtype)
    return image_numpy


def _flag_enabled(data, key, default=False):
    value = data.get(key, default)
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    if isinstance(value, (list, tuple)):
        return any(bool(v) for v in value)
    return bool(value)


class EdgeMap(nn.Module):
    def __init__(self, scale=1):
        super(EdgeMap, self).__init__()
        self.scale = scale
        self.requires_grad = False

    def forward(self, img):
        img = img / self.scale

        N, C, H, W = img.shape
        gradX = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        
        gradx = (img[...,1:,:] - img[...,:-1,:]).abs().sum(dim=1, keepdim=True)
        grady = (img[...,1:] - img[...,:-1]).abs().sum(dim=1, keepdim=True)

        gradX[...,:-1,:] += gradx
        gradX[...,1:,:] += gradx
        gradX[...,1:-1,:] /= 2

        gradY[...,:-1] += grady
        gradY[...,1:] += grady
        gradY[...,1:-1] /= 2

        # edge = (gradX + gradY) / 2
        edge = (gradX + gradY)

        return edge


class ERRNetBase(BaseModel):
    def _init_optimizer(self, optimizers):
        self.optimizers = optimizers
        for optimizer in self.optimizers:
            util.set_opt_param(optimizer, 'initial_lr', self.opt.lr)
            util.set_opt_param(optimizer, 'weight_decay', self.opt.wd)

    def set_input(self, data, mode='train'):
        target_t = None
        target_r = None
        data_name = None
        mode = mode.lower()
        if mode == 'train':
            input, target_t, target_r = data['input'], data['target_t'], data['target_r']
        elif mode == 'eval':
            input, target_t, target_r, data_name = data['input'], data['target_t'], data['target_r'], data['fn']
        elif mode == 'test':
            input, data_name = data['input'], data['fn']
        else:
            raise NotImplementedError('Mode [%s] is not implemented' % mode)
        
        if len(self.gpu_ids) > 0:  # transfer data into gpu
            input = input.to(self.device, non_blocking=True)
            if target_t is not None:
                target_t = target_t.to(self.device, non_blocking=True)
            if target_r is not None:
                target_r = target_r.to(self.device, non_blocking=True)
        
        self.input = input
        
        self.input_edge = self.edge_map(self.input)
        self.target_t = target_t
        self.data_name = data_name

        self.issyn = not _flag_enabled(data, 'real', default=False)
        self.aligned = not _flag_enabled(data, 'unaligned', default=False)
        
        if target_t is not None:            
            self.target_edge = self.edge_map(self.target_t)         

    def _using_ema_for_eval(self):
        return False

    @contextmanager
    def _ema_eval_context(self):
        yield
	            
    def eval(self, data, savedir=None, suffix=None, pieapp=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'eval')

        with torch.no_grad():
            with self._ema_eval_context():
                self.forward()

            output_i = tensor2im(self.output_i)
            target = tensor2im(self.target_t)

            if self.aligned:
                h = min(output_i.shape[0], target.shape[0])
                w = min(output_i.shape[1], target.shape[1])
                res = index.quality_assess(output_i[:h, :w], target[:h, :w])
            else:
                res = {}

            if savedir is not None:
                if self.data_name is not None:
                    name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
                    if not os.path.exists(join(savedir, name)):
                        os.makedirs(join(savedir, name))
                    if suffix is not None:
                        Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name,'{}_{}.png'.format(self.opt.name, suffix)))
                    else:
                        Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name, '{}.png'.format(self.opt.name)))
                    Image.fromarray(target.astype(np.uint8)).save(join(savedir, name, 't_label.png'))
                    Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, name, 'm_input.png'))
                else:
                    if not os.path.exists(join(savedir, 'transmission_layer')):
                        os.makedirs(join(savedir, 'transmission_layer'))
                        os.makedirs(join(savedir, 'blended'))
                    Image.fromarray(target.astype(np.uint8)).save(join(savedir, 'transmission_layer', str(self._count)+'.png'))
                    Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, 'blended', str(self._count)+'.png'))
                    self._count += 1

            return res

    def test(self, data, savedir=None):
        # only the 1st input of the whole minibatch would be evaluated
        self._eval()
        self.set_input(data, 'test')

        if self.data_name is not None and savedir is not None:
            name = os.path.splitext(os.path.basename(self.data_name[0]))[0]
            if not os.path.exists(join(savedir, name)):
                os.makedirs(join(savedir, name))
        
        with torch.no_grad():
            with self._ema_eval_context():
                output_i = self.forward()
            output_i = tensor2im(output_i)
                # if os.path.exists(join(savedir, name,'t_output.png')):
                #     i = 2
                #     while True:
                #         if not os.path.exists(join(savedir, name,'t_output_{}.png'.format(i))):
                #             Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name,'t_output_{}.png'.format(i)))
                #             break
                #         i += 1
                # else:
                #     Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name,'t_output.png'))
            if self.data_name is not None and savedir is not None:                
                Image.fromarray(output_i.astype(np.uint8)).save(join(savedir, name, '{}.png'.format(self.opt.name)))
                Image.fromarray(tensor2im(self.input).astype(np.uint8)).save(join(savedir, name, 'm_input.png'))


class ERRNetModel(ERRNetBase):
    def name(self):
        return 'errnet'
        
    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cpu")
        self.ema_state = None
        self._icnn_partial_load = False

    def print_network(self):
        print('--------------------- Model ---------------------')
        print('##################### NetG #####################')
        networks.print_network(self.net_i)
        if self.isTrain and self.opt.lambda_gan > 0:
            print('##################### NetD #####################')
            networks.print_network(self.netD)

    def _eval(self):
        self.net_i.eval()

    def _train(self):
        self.net_i.train()

    def _net_i_module(self):
        return _unwrap_ddp(self.net_i)

    def _ema_enabled(self):
        return getattr(self.opt, 'ema_decay', 0) > 0

    def _init_ema_state(self):
        self.ema_state = _copy_state_dict(self._net_i_module().state_dict())

    def _load_ema_state(self, state_dict):
        cleaned = _clean_state_dict(state_dict)
        if getattr(self.opt, 'hyper', False):
            cleaned, _, _ = self._compatible_icnn_state(cleaned, prefix='icnn_ema')
            ema_state = _copy_state_dict(self._net_i_module().state_dict(), device=self.device)
            ema_state.update(_copy_state_dict(cleaned, device=self.device))
            self.ema_state = ema_state
        else:
            self.ema_state = _copy_state_dict(cleaned, device=self.device)

    def _compatible_icnn_state(self, state_dict, prefix='icnn'):
        target_state = self._net_i_module().state_dict()
        compatible = {}
        converted = []
        skipped = []

        for key, value in state_dict.items():
            if key not in target_state:
                skipped.append(key)
                continue

            target_value = target_state[key]
            if value.shape == target_value.shape:
                compatible[key] = value
                continue

            if (
                key == 'conv1.conv2d.weight'
                and value.dim() == 4
                and target_value.dim() == 4
                and target_value.shape[1] >= 3
                and value.shape[1] >= 3
                and value.shape[0] == target_value.shape[0]
                and value.shape[2:] == target_value.shape[2:]
            ):
                converted_value = target_value.clone()
                converted_value[:, :3] = value[:, :3]
                compatible[key] = converted_value.contiguous()
                converted.append(key)
                continue

            skipped.append(key)

        missing = [key for key in target_state.keys() if key not in compatible]
        if converted:
            print('[i] {}: converted old hyper input weights for {}'.format(prefix, ', '.join(converted)))
        if missing or skipped:
            print('[i] {}: partial load, missing {}, skipped {}'.format(prefix, len(missing), len(skipped)))
        return compatible, missing, skipped

    def _load_icnn_state(self, state_dict):
        cleaned = _clean_state_dict(state_dict)
        if getattr(self.opt, 'hyper', False):
            compatible, _, _ = self._compatible_icnn_state(cleaned, prefix='icnn')
            self._net_i_module().load_state_dict(compatible, strict=False)
            self._icnn_partial_load = len(compatible) != len(self._net_i_module().state_dict())
        else:
            self.net_i.load_state_dict(cleaned)
            self._icnn_partial_load = False

    def _update_ema(self):
        if not self._ema_enabled():
            return
        module_state = self._net_i_module().state_dict()
        if self.ema_state is None:
            self._init_ema_state()
            return

        decay = self.opt.ema_decay
        for key, value in module_state.items():
            value = value.detach()
            if key not in self.ema_state:
                self.ema_state[key] = value.clone()
            elif torch.is_floating_point(value):
                self.ema_state[key].mul_(decay).add_(value, alpha=1.0 - decay)
            else:
                self.ema_state[key].copy_(value)

    def _using_ema_for_eval(self):
        return self.ema_state is not None and not getattr(self.opt, 'no_ema_eval', False)

    @contextmanager
    def _ema_eval_context(self):
        if not self._using_ema_for_eval():
            yield
            return

        module = self._net_i_module()
        backup = _copy_state_dict(module.state_dict())
        module.load_state_dict(self.ema_state)
        try:
            yield
        finally:
            module.load_state_dict(backup)

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.device = opt.device

        in_channels = 3
        dino_channels = None
        self.feature_extractor = None
        self.vgg_feature_extractor = None
        feature_layers = [int(layer) for layer in str(opt.feature_layers).split(',') if layer.strip()]
        
        if opt.hyper:
            self.vgg_feature_extractor = losses.Vgg19(requires_grad=False).to(self.device)
            self.vgg_feature_extractor.eval()
            in_channels += 64 + 128 + 256 + 512 + 512
            self.feature_extractor = losses.DINOv3Features(
                model_path=opt.feature_model_path,
                layers=feature_layers,
                feature_scale=opt.feature_scale,
                normalize_features=not opt.no_feature_norm,
                requires_grad=False,
            ).to(self.device)
            dino_channels = self.feature_extractor.out_channels
        
        self.net_i = arch.__dict__[self.opt.inet](
            in_channels, 3,
            dino_channels=dino_channels,
            fusion_type=opt.fusion_type,
            full_res=not opt.no_full_res,
        ).to(self.device)
        networks.init_weights(self.net_i, init_type=opt.init_type) # using default initialization as EDSR
        self.edge_map = EdgeMap(scale=1).to(self.device)
        self.ema_state = None

        if self.isTrain:
            self.loss_feature_extractor = losses.Vgg19(requires_grad=False).to(self.device)

            # define loss functions
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            vggloss = losses.ContentLoss()
            vggloss.initialize(losses.VGGLoss(self.loss_feature_extractor))
            self.loss_dic['t_vgg'] = vggloss

            cxloss = losses.ContentLoss()
            if opt.unaligned_loss == 'vgg':
                cxloss.initialize(losses.VGGLoss(self.loss_feature_extractor, weights=[0.1], indices=[opt.vgg_layer]))
            elif opt.unaligned_loss == 'ctx':
                cxloss.initialize(losses.CXLoss(self.loss_feature_extractor, weights=[0.1,0.1,0.1], indices=[8, 13, 22]))
            elif opt.unaligned_loss == 'mse':
                cxloss.initialize(nn.MSELoss())
            elif opt.unaligned_loss == 'ctx_vgg':
                cxloss.initialize(losses.CXLoss(self.loss_feature_extractor, weights=[0.1,0.1,0.1,0.1], indices=[8, 13, 22, 31], criterions=[losses.CX_loss]*3+[nn.L1Loss()]))
            else:
                raise NotImplementedError

            self.loss_dic['t_cx'] = cxloss

            # Define discriminator
            # if self.opt.lambda_gan > 0:
            self.netD = networks.define_D(opt, 3).to(self.device)
            self.optimizer_D = _build_optimizer(opt, self.netD.parameters())

            # initialize optimizers
            self.optimizer_G = _build_optimizer(opt, self.net_i.parameters())

            self._init_optimizer([self.optimizer_G, self.optimizer_D])

        if opt.resume:
            self.load(self, opt.resume_epoch)

        if self.isTrain and self._ema_enabled() and self.ema_state is None:
            self._init_ema_state()

        if opt.distributed:
            self.net_i = DistributedDataParallel(
                self.net_i,
                device_ids=[opt.local_rank],
                output_device=opt.local_rank,
                find_unused_parameters=False,
            )
            if self.isTrain:
                self.netD = DistributedDataParallel(
                    self.netD,
                    device_ids=[opt.local_rank],
                    output_device=opt.local_rank,
                    find_unused_parameters=False,
                )
        
        if opt.no_verbose is False:
            self.print_network()

    def backward_D(self):
        for p in self.netD.parameters():
            p.requires_grad = True

        self.loss_D, self.pred_fake, self.pred_real = self.loss_dic['gan'].get_loss(
            self.netD, self.input, self.output_i, self.target_t)

        (self.loss_D*self.opt.lambda_gan).backward(retain_graph=True)

    def backward_G(self):
        # Make it a tiny bit faster
        for p in self.netD.parameters():
            p.requires_grad = False
        
        self.loss_G = 0
        self.loss_CX = None
        self.loss_icnn_pixel = None
        self.loss_icnn_vgg = None
        self.loss_G_GAN = None

        if self.opt.lambda_gan > 0:
            self.loss_G_GAN = self.loss_dic['gan'].get_g_loss(
                self.netD, self.input, self.output_i, self.target_t) #self.pred_real.detach())
            self.loss_G += self.loss_G_GAN*self.opt.lambda_gan
        
        if self.aligned:
            self.loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(
                self.output_i, self.target_t)
            
            self.loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(
                self.output_i, self.target_t)

            self.loss_G += self.loss_icnn_pixel+self.loss_icnn_vgg*self.opt.lambda_vgg
        else:
            self.loss_CX = self.loss_dic['t_cx'].get_loss(self.output_i, self.target_t)
            
            self.loss_G += self.loss_CX
        
        self.loss_G.backward()

    def forward(self):
        # without edge
        input_i = self.input
        dino_feature = None

        if self.feature_extractor is not None:
            _, _, height, width = self.input.shape
            with torch.no_grad():
                vgg_features = self.vgg_feature_extractor(self.input)
                vgg_features = [
                    F.interpolate(feature, size=(height, width), mode='bilinear', align_corners=False)
                    for feature in vgg_features
                ]
                dino_feature = self.feature_extractor(self.input, [self.opt.hyper_layer])[0]
            input_i = torch.cat([self.input] + vgg_features, dim=1)
            dino_feature = dino_feature.detach()

        net_i = self.net_i
        if isinstance(net_i, DistributedDataParallel) and not net_i.training:
            net_i = _unwrap_ddp(net_i)

        output_i = net_i(input_i, dino_feature)

        self.output_i = output_i

        return output_i
        
    def optimize_parameters(self):
        self._train()
        self.forward()

        if self.opt.lambda_gan > 0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            if self.opt.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.netD.parameters(), self.opt.clip_grad_norm)
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        if self.opt.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.net_i.parameters(), self.opt.clip_grad_norm)
        self.optimizer_G.step()
        self._update_ema()
        
    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_icnn_pixel is not None:
            ret_errors['IPixel'] = self.loss_icnn_pixel.item()
        if self.loss_icnn_vgg is not None:
            ret_errors['VGG'] = self.loss_icnn_vgg.item()
            
        if self.opt.lambda_gan > 0 and self.loss_G_GAN is not None:
            ret_errors['G'] = self.loss_G_GAN.item()
            ret_errors['D'] = self.loss_D.item()

        if self.loss_CX is not None:
            ret_errors['CX'] = self.loss_CX.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input).astype(np.uint8)
        ret_visuals['output_i'] = tensor2im(self.output_i).astype(np.uint8)        
        ret_visuals['target'] = tensor2im(self.target_t).astype(np.uint8)
        ret_visuals['residual'] = tensor2im((self.input - self.output_i)).astype(np.uint8)

        return ret_visuals       

    @staticmethod
    def load(model, resume_epoch=None):
        icnn_path = model.opt.icnn_path
        state_dict = None

        if icnn_path is None:
            model_path = util.get_model_list(model.save_dir, model.name(), epoch=resume_epoch)
            state_dict = _torch_load_compat(model_path)
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']
            model._load_icnn_state(state_dict['icnn'])
            if 'icnn_ema' in state_dict:
                model._load_ema_state(state_dict['icnn_ema'])
            if model.isTrain and not model._icnn_partial_load:
                model.optimizer_G.load_state_dict(state_dict['opt_g'])
            elif model.isTrain:
                print('[i] skip optimizer_G state because icnn was partially loaded')
        else:
            state_dict = _torch_load_compat(icnn_path, map_location=torch.device('cpu'))
            model._load_icnn_state(state_dict['icnn'])
            if 'icnn_ema' in state_dict:
                model._load_ema_state(state_dict['icnn_ema'])
            model.epoch = state_dict['epoch']
            model.iterations = state_dict['iterations']
            # if model.isTrain:
            #     model.optimizer_G.load_state_dict(state_dict['opt_g'])

        if model.isTrain:
            if 'netD' in state_dict:
                print('Resume netD ...')
                model.netD.load_state_dict(_clean_state_dict(state_dict['netD']))
                model.optimizer_D.load_state_dict(state_dict['opt_d'])
            
        print('Resume from epoch %d, iteration %d' % (model.epoch, model.iterations))
        return state_dict

    def state_dict(self):
        state_dict = {
            'icnn': _unwrap_ddp(self.net_i).state_dict(),
            'opt_g': self.optimizer_G.state_dict(), 
            'epoch': self.epoch, 'iterations': self.iterations
        }

        if self.ema_state is not None:
            state_dict['icnn_ema'] = self.ema_state

        if self.opt.lambda_gan > 0:
            state_dict.update({
                'opt_d': self.optimizer_D.state_dict(),
                'netD': _unwrap_ddp(self.netD).state_dict(),
            })

        return state_dict


class NetworkWrapper(ERRNetBase):
    # You can use this class to wrap other module into our training framework (\eg BDN module)
    def __init__(self):
        self.epoch = 0
        self.iterations = 0
        self.device = torch.device("cpu")

    def print_network(self):
        print('--------------------- NetworkWrapper ---------------------')
        networks.print_network(self.net)

    def _eval(self):
        self.net.eval()

    def _train(self):
        self.net.train()

    def initialize(self, opt, net):
        BaseModel.initialize(self, opt)
        self.device = torch.device("cuda:%d" % self.gpu_ids[0] if len(self.gpu_ids) > 0 else "cpu")
        self.net = net.to(self.device)
        self.edge_map = EdgeMap(scale=1).to(self.device)
        
        if self.isTrain:
            # define loss functions
            self.loss_feature_extractor = losses.Vgg19(requires_grad=False).to(self.device)
            self.loss_dic = losses.init_loss(opt, self.Tensor)
            vggloss = losses.ContentLoss()
            vggloss.initialize(losses.VGGLoss(self.loss_feature_extractor))
            self.loss_dic['t_vgg'] = vggloss

            cxloss = losses.ContentLoss()
            if opt.unaligned_loss == 'vgg':
                cxloss.initialize(losses.VGGLoss(self.loss_feature_extractor, weights=[0.1], indices=[opt.vgg_layer]))
            elif opt.unaligned_loss == 'ctx':
                cxloss.initialize(losses.CXLoss(self.loss_feature_extractor, weights=[0.1,0.1,0.1], indices=[8, 13, 22]))
            elif opt.unaligned_loss == 'mse':
                cxloss.initialize(nn.MSELoss())
            elif opt.unaligned_loss == 'ctx_vgg':
                cxloss.initialize(losses.CXLoss(self.loss_feature_extractor, weights=[0.1,0.1,0.1,0.1], indices=[8, 13, 22, 31], criterions=[losses.CX_loss]*3+[nn.L1Loss()]))
                
            else:
                raise NotImplementedError            
            
            self.loss_dic['t_cx'] = cxloss

            # initialize optimizers
            self.optimizer_G = _build_optimizer(opt, self.net.parameters())

            self._init_optimizer([self.optimizer_G])

            # define discriminator
            # if self.opt.lambda_gan > 0:
            self.netD = networks.define_D(opt, 3)
            self.optimizer_D = _build_optimizer(opt, self.netD.parameters())
            self._init_optimizer([self.optimizer_D])
        
        if opt.no_verbose is False:
            self.print_network()

    def backward_D(self):
        for p in self.netD.parameters():
            p.requires_grad = True

        self.loss_D, self.pred_fake, self.pred_real = self.loss_dic['gan'].get_loss(
            self.netD, self.input, self.output_i, self.target_t)

        (self.loss_D*self.opt.lambda_gan).backward(retain_graph=True)
        
    def backward_G(self):
        for p in self.netD.parameters():
            p.requires_grad = False
                    
        self.loss_G = 0
        self.loss_CX = None
        self.loss_icnn_pixel = None
        self.loss_icnn_vgg = None
        self.loss_G_GAN = None

        if self.opt.lambda_gan > 0:
            self.loss_G_GAN = self.loss_dic['gan'].get_g_loss(
                self.netD, self.input, self.output_i, self.target_t) #self.pred_real.detach())
            self.loss_G += self.loss_G_GAN*self.opt.lambda_gan
                
        if self.aligned:
            self.loss_icnn_pixel = self.loss_dic['t_pixel'].get_loss(
                self.output_i, self.target_t)
            
            self.loss_icnn_vgg = self.loss_dic['t_vgg'].get_loss(
                self.output_i, self.target_t)

            # self.loss_G += self.loss_icnn_pixel
            self.loss_G += self.loss_icnn_pixel+self.loss_icnn_vgg*self.opt.lambda_vgg
            # self.loss_G += self.loss_fm * self.opt.lambda_vgg
        else:
            self.loss_CX = self.loss_dic['t_cx'].get_loss(self.output_i, self.target_t)
            
            self.loss_G += self.loss_CX
        
        self.loss_G.backward()

    def forward(self):
        raise NotImplementedError
        
    def optimize_parameters(self):
        self._train()
        self.forward()

        if self.opt.lambda_gan > 0:
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
    def get_current_errors(self):
        ret_errors = OrderedDict()
        if self.loss_icnn_pixel is not None:
            ret_errors['IPixel'] = self.loss_icnn_pixel.item()
        if self.loss_icnn_vgg is not None:
            ret_errors['VGG'] = self.loss_icnn_vgg.item()
        if self.opt.lambda_gan > 0 and self.loss_G_GAN is not None:
            ret_errors['G'] = self.loss_G_GAN.item()
            ret_errors['D'] = self.loss_D.item()
        if self.loss_CX is not None:
            ret_errors['CX'] = self.loss_CX.item()

        return ret_errors

    def get_current_visuals(self):
        ret_visuals = OrderedDict()
        ret_visuals['input'] = tensor2im(self.input).astype(np.uint8)
        ret_visuals['output_i'] = tensor2im(self.output_i).astype(np.uint8)        
        ret_visuals['target'] = tensor2im(self.target_t).astype(np.uint8)
        ret_visuals['residual'] = tensor2im((self.input - self.output_i)).astype(np.uint8)
        return ret_visuals

    def state_dict(self):
        state_dict = self.net.state_dict()
        return state_dict
