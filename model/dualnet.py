# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.autograd import Variable
from .common import MLP, ResNet18
from .common import MaskNet18

import pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kornia.augmentation as K
import kornia

from copy import deepcopy
from torchvision import datasets, transforms
class Net(torch.nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        self.reg = args.memory_strength
        self.temp = args.temperature
        self.debugger = 0
        # setup network
        self.is_cifar = any(x in str(args.data_file) for x in ['cifar', 'cub', 'mini'])
        
        self.is_cifar=True
        
        nf = 64 if 'core' in args.train_csv else 64
        lr_ = 1e-4 if 'core' in args.train_csv else 3e-4
        n_outputs = 50 if 'core' in args.train_csv else 85
        self.net = MaskNet18(n_outputs, nf=nf)
        
        self.lr = args.lr
        self.transforms1 = nn.Sequential(
                K.RandomCrop((84,84)), K.RandomHorizontalFlip(),
                K.ColorJitter(brightness=0.4, contrast=0.4,saturation=0.2, hue=0.1, p=0.8),
                K.RandomGrayscale(p=0.2),
                K.GaussianBlur((3,3),(0.1,2.0), p=1.0),
                #K.RandomSolarize(p=0.0),
                K.Normalize(torch.FloatTensor((0.5,0.5,0.5)), torch.FloatTensor((0.5,0.5,0.5))))
        self.transforms2 = nn.Sequential(
                K.RandomCrop((84,84)), K.RandomHorizontalFlip(),
                K.ColorJitter(brightness=0.4, contrast=0.4,saturation=0.2, hue=0.1, p=0.8),
                K.RandomGrayscale(p=0.2),
                K.GaussianBlur((3,3),(0.1,2.0), p=0.1),
                #K.RandomSolarize(p=0.2),
                K.Normalize(torch.FloatTensor((0.5,0.5,0.5)), torch.FloatTensor((0.5,0.5,0.5))))
        self.transforms0 = nn.Sequential(
                transforms.Resize((84,84)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))
                #K.Normalize(torch.FloatTensor((0.5,0.5,0.5)), torch.FloatTensor((0.5,0.5,0.5))))

        self.transforms0 = torch.nn.DataParallel(self.transforms0, [torch.cuda.device(0), torch.cuda.device(1)])
        self.transforms1 = torch.nn.DataParallel(self.transforms1, [torch.cuda.device(0), torch.cuda.device(1)])
        self.transforms2 = torch.nn.DataParallel(self.transforms2, [torch.cuda.device(0), torch.cuda.device(1)])
        self.beta = args.beta
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        
        self.bt_opt = torch.optim.SGD(self.net.parameters(), lr=lr_)
        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()

        if self.is_cifar:
            self.nc_per_task = int(n_outputs // n_tasks)
        else:
            self.nc_per_task = n_outputs
        # setup memories
        self.current_task = 0
        self.fisher = {}
        self.optpar = {}
        self.n_memories = args.n_memories
        self.mem_cnt = 0
        self.memx = torch.FloatTensor(n_tasks, self.n_memories, 3, 128, 128)
        self.memy = torch.LongTensor(n_tasks, self.n_memories)
        self.mem_feat = torch.FloatTensor(n_tasks, self.n_memories, self.nc_per_task)
        self.mem = {}
        if args.cuda:
            self.memx = self.memx.cuda()
            self.memy = self.memy.cuda()
            self.mem_feat = self.mem_feat.cuda()
        self.mem_cnt = 0
        self.n_memories = args.n_memories
        self.bsz = args.batch_size
        
        self.n_outputs = n_outputs

        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss()
        self.samples_seen = 0
        self.samples_per_task = args.samples_per_task
        self.sz = args.replay_batch_size
        self.inner_steps = args.inner_steps
        self.n_outer = args.n_outer
    def on_epoch_end(self):  
        pass

    def compute_offsets(self, task):
        if self.is_cifar:
            offset1 = task * self.nc_per_task
            offset2 = (task + 1) * self.nc_per_task
        else:
            offset1 = 0
            offset2 = self.n_outputs
        return int(offset1), int(offset2)

    def forward(self, x, t, return_feat= False):
        if not self.training:
            x = self.transforms0(x).cuda()
        output = self.net(x)    
        #if self.is_cifar:
            # make sure we predict classes within the current task
        offset1, offset2 = self.compute_offsets(t)

        if offset1 > 0:
            output[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_outputs:
            output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output
    
    def memory_sampling(self,t, sz_=-1):
        mem_x = self.memx[:t,:]
        mem_y = self.memy[:t,:]
        mem_feat = self.mem_feat[:t,:]
        
        sz = min(self.n_memories, self.sz) if sz_ < 0 else sz_
        idx = np.random.choice(t* self.n_memories,sz, False)
        t_idx = torch.from_numpy(idx // self.n_memories)
        s_idx = torch.from_numpy( idx % self.n_memories)

        offsets = torch.tensor([self.compute_offsets(i) for i in t_idx]).cuda()
        xx = mem_x[t_idx, s_idx]
        yy = mem_y[t_idx, s_idx] - offsets[:,0]
        feat = mem_feat[t_idx, s_idx]
        mask = torch.zeros(xx.size(0), self.nc_per_task)
        for j in range(mask.size(0)):
            mask[j] = torch.arange(offsets[j][0], offsets[j][1])
        return xx,yy, feat , mask.long().cuda()
    def observe(self, x, t, y):
        
        self.debugger += 1
        if t != self.current_task:
            tt = self.current_task
            offset1, offset2 = self.compute_offsets(tt)
            x_old = self.transforms0(self.memx[tt]).cuda()
            out = self.forward(x_old, tt, True)
            self.mem_feat[tt] = F.softmax(out[:, offset1:offset2] / self.temp, dim=1 ).data.clone()
            self.current_task = t       #idx = info[1]
        self.net.train()
        bsz = y.data.size(0)
        endcnt = min(self.mem_cnt + bsz, self.n_memories)
        effbsz = endcnt - self.mem_cnt
        
        self.memx[t, self.mem_cnt: endcnt].copy_(x.data[: effbsz])
        self.memy[t, self.mem_cnt: endcnt].copy_(y.data[: effbsz])
        self.mem_cnt += effbsz
        if self.mem_cnt == self.n_memories:
            self.mem_cnt = 0
        
        for j in range(self.n_outer):
            weights_before = deepcopy(self.net.state_dict())
            for _ in range(self.inner_steps):
                self.zero_grad()
                if t > 0:
                    xx, yy, target, mask = self.memory_sampling(t, 32)
                    x1, x2 = self.transforms1(xx), self.transforms2(xx)
                else:
                    x1, x2= self.transforms1(x), self.transforms2(x)
                loss0 = self.net.BarlowTwins(x1,x2)
                loss0.backward()
                self.bt_opt.step()
            weights_after = self.net.state_dict()
            new_params = {name : weights_before[name] + ((weights_after[name] - weights_before[name]) * self.beta) for name in weights_before.keys()}
            self.net.load_state_dict(new_params)
        
        for _ in range(self.inner_steps):
            self.zero_grad()
            loss1 = torch.tensor(0.).cuda()
            loss2 = torch.tensor(0.).cuda()
            loss3 = torch.tensor(0.).cuda()
            x_ = self.transforms0(x).cuda()
            
            offset1, offset2 = self.compute_offsets(t)
            pred = self.forward(x_,t, True)
            loss1 = self.bce(pred[:, offset1:offset2], y - offset1)
            if t > 0:
                xx, yy, target, mask = self.memory_sampling(t)
                xx = self.transforms0(xx).cuda()
                pred_ = self.net(xx)
                pred = torch.gather(pred_, 1, mask)
                loss2 += self.bce(pred, yy)
                loss3 = self.reg * self.kl(F.log_softmax(pred / self.temp, dim = 1), target)       
            loss = loss1 + loss2 + loss3
            loss.backward()
            self.opt.step()
        return loss.item()
