import torch
import torch.distributed as dist
import util.util as util
import models
import time
import os
import sys
from os.path import join
from util.visualizer import Visualizer
from util.distributed import barrier, is_main_process


def _reduce_average_meters(avg_meters, opt):
    if not getattr(opt, "distributed", False):
        return avg_meters

    gathered_keys = [None for _ in range(opt.world_size)]
    dist.all_gather_object(gathered_keys, list(avg_meters.keys()))
    keys = sorted({key for rank_keys in gathered_keys for key in rank_keys})

    reduced = util.AverageMeters()
    for key in keys:
        value = avg_meters.dic.get(key, 0.0)
        count = avg_meters.total_num.get(key, 0)
        stats = torch.tensor([float(value), float(count)], dtype=torch.float64, device=opt.device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if stats[1].item() > 0:
            reduced.dic[key] = stats[0].item()
            reduced.total_num[key] = int(stats[1].item())
    return reduced


class Engine(object):
    def __init__(self, opt):
        self.opt = opt
        self.writer = None
        self.visualizer = None
        self.model = None
        self.best_val_loss = 1e6

        self.__setup()

    def __setup(self):
        self.basedir = join('checkpoints', self.opt.name)
        if is_main_process(self.opt) and not os.path.exists(self.basedir):
            os.mkdir(self.basedir)
        
        opt = self.opt
        
        """Model"""
        self.model = models.__dict__[self.opt.model]()
        self.model.initialize(opt)
        if not opt.no_log and is_main_process(opt):
            self.writer = util.get_summary_writer(os.path.join(self.basedir, 'logs'))
            self.visualizer = Visualizer(opt)

    def train(self, train_loader, **kwargs):
        opt = self.opt
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(self.epoch)

        if is_main_process(opt):
            print('\nEpoch: %d' % self.epoch)
        avg_meters = util.AverageMeters()
        model = self.model
        epoch = self.epoch

        epoch_start_time = time.time()
        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            iterations = self.iterations
            

            model.set_input(data, mode='train')
            model.optimize_parameters(**kwargs)
            
            errors = model.get_current_errors()
            avg_meters.update(errors)
            if is_main_process(opt):
                util.progress_bar(i, len(train_loader), str(avg_meters))
            
            if not opt.no_log and is_main_process(opt):
                util.write_loss(self.writer, 'train', avg_meters, iterations)
            
                if iterations % opt.display_freq == 0 and opt.display_id != 0:
                    save_result = iterations % opt.update_html_freq == 0
                    self.visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if iterations % opt.print_freq == 0 and opt.display_id != 0:
                    t = (time.time() - iter_start_time)          

            self.iterations += 1
    
        self.epoch += 1

        if not self.opt.no_log and is_main_process(opt):
            if self.epoch % opt.save_epoch_freq == 0:
                print('saving the model at epoch %d, iters %d' %
                    (self.epoch, self.iterations))
                model.save()
            
            print('saving the latest model at the end of epoch %d, iters %d' % 
                (self.epoch, self.iterations))
            model.save(label='latest')

            print('Time Taken: %d sec' %
                (time.time() - epoch_start_time))
                
        # model.update_learning_rate()
        if hasattr(train_loader, "reset"):
            train_loader.reset()
        barrier(opt)

    def eval(self, val_loader, dataset_name, savedir=None, loss_key=None, sync_distributed=False, **kwargs):
        
        avg_meters = util.AverageMeters()
        model = self.model
        opt = self.opt
        with torch.no_grad():
            for i, data in enumerate(val_loader):                
                index = model.eval(data, savedir=savedir, **kwargs)
                avg_meters.update(index)
                
                if is_main_process(opt):
                    util.progress_bar(i, len(val_loader), str(avg_meters))
                
        if sync_distributed:
            avg_meters = _reduce_average_meters(avg_meters, opt)
            barrier(opt)

        if not opt.no_log and is_main_process(opt):
            util.write_loss(self.writer, join('eval', dataset_name), avg_meters, self.epoch)
        
        if loss_key is not None:
            val_loss = avg_meters[loss_key]
            if is_main_process(opt) and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                print('saving the best model at the end of epoch %d, iters %d' % 
                    (self.epoch, self.iterations))
                model.save(label='best_{}_{}'.format(loss_key, dataset_name))

        return avg_meters

    def test(self, test_loader, savedir=None, **kwargs):
        model = self.model
        opt = self.opt
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                model.test(data, savedir=savedir, **kwargs)
                if is_main_process(opt):
                    util.progress_bar(i, len(test_loader))
        barrier(opt)

    @property
    def iterations(self):
        return self.model.iterations

    @iterations.setter
    def iterations(self, i):
        self.model.iterations = i

    @property
    def epoch(self):
        return self.model.epoch

    @epoch.setter
    def epoch(self, e):
        self.model.epoch = e
