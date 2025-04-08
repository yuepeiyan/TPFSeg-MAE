import math
import os
import shutil

import torch
import torch.nn as nn

from .. import models
from .. import networks


class BaseTrainer(object):
    r"""
    Base class for all the trainers
    """

    def __init__(self, args):
        self.args = args
        self.model_name = 'Unknown'
        self.model = None
        self.wrapped_model = None
        self.optimizer = None
        self.dataloader = None
        # The following attributes will be modiflied adaptively
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.init_lr()  # lr rate
        # create checkpoint directory
        if args.rank == 0:
            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)

    def init_lr(self):
        args = self.args
        # infer learning rate before changing batch size
        self.lr = args.lr * args.batch_size / 256

    def build_model(self):
        if self.model_name != 'Unknown' and self.model is None:
            args = self.args
            print(f"=> creating model {self.model_name} of arch {args.arch}")
            self.model = getattr(models, self.model_name)(getattr(networks, args.arch), args)
            self.wrap_model()
        elif self.model_name == 'Unknown':
            raise ValueError("=> Model name is still unknown")
        else:
            raise ValueError("=> Model has been created. Do not create twice")

    def build_optimizer(self):
        assert (
                    self.model is not None and self.wrapped_model is not None), "Model is not created and wrapped yet. Please create model first."
        print("=> creating optimizer")
        args = self.args
        model = self.wrapped_model

        optim_params = self.group_params(model)
        # TODO: create optimizer factory
        self.optimizer = torch.optim.SGD(optim_params, self.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

    def wrap_model(self):
        """
        1. Distribute model or not
        2. Rewriting batch size and workers
        """
        args = self.args
        model = self.model
        assert model is not None, "Please build model before wrapping model"

        if args.distributed:
            ngpus_per_node = args.ngpus_per_node
            # Apply SyncBN
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.batch_size = args.batch_size // ngpus_per_node
                self.workers = (args.workers + ngpus_per_node - 1) // ngpus_per_node
                print("=> Finish adapting batch size and workers according to gpu number")
                model = nn.parallel.DistributedDataParallel(model,
                                                            device_ids=[args.gpu],
                                                            find_unused_parameters=True)
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            # AllGather/rank implementation in this code only supports DistributedDataParallel
            raise NotImplementedError("Must Specify GPU or use DistributeDataParallel.")

        self.wrapped_model = model

    def group_params(self, model, fix_lr=False):
        all_params = set(model.parameters())
        wd_params = set()
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                wd_params.add(m.weight)
        no_wd = all_params - wd_params
        params_group = [{'params': list(wd_params), 'fix_lr': fix_lr},
                        {'params': list(no_wd), 'weight_decay': 0., 'fix_lr': fix_lr}]
        return params_group

    def get_parameter_groups(self, get_layer_id=None, get_layer_scale=None, verbose=False):
        args = self.args
        weight_decay = args.weight_decay
        model = self.model

        if hasattr(model, 'no_weight_decay'):
            skip_list = model.no_weight_decay()
        else:
            skip_list = {}

        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
            if get_layer_id is not None:
                layer_id = get_layer_id(name)
                group_name = "layer_%d_%s" % (layer_id, group_name)
            else:
                layer_id = None

            if group_name not in parameter_group_names:
                if get_layer_scale is not None:
                    scale = get_layer_scale(layer_id)
                else:
                    scale = 1.

                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }

            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
        if verbose:
            import json
            print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        else:
            print("Param groups information is omitted...")
        return list(parameter_group_vars.values())

    def resume(self):
        args = self.args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']
            self.model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    def run(self, train_sampler, ngpus_per_node):
        args = self.args
        # Compute iterations when resuming
        niters = args.start_epoch * self.iters_per_epoch

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                self.dataloader.sampler.set_epoch(epoch)

            # train for one epoch
            niters = self.epoch_train(epoch, niters)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                if epoch == 0 or (epoch + 1) % args.save_freq == 0:
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }, is_best=False, filename=f'{args.ckpt_dir}/checkpoint_{epoch:04d}.pth.tar')

    def epoch_train(self, train_loader, model, optimizer, epoch, niters, args):
        raise NotImplementedError("train method for base class Trainer has not been implemented yet.")

    def adjust_learning_rate(self, epoch, args):
        """Base schedule: CosineDecay with warm-up."""
        init_lr = self.lr
        if epoch < args.warmup_epochs:
            cur_lr = init_lr * epoch / args.warmup_epochs
        else:
            cur_lr = init_lr * 0.5 * (
                        1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = init_lr
            else:
                param_group['lr'] = cur_lr
