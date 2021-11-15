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

kl = lambda y, t_s, t : F.kl_div(F.log_softmax(y / t, dim=-1), F.softmax(t_s / t, dim=-1), reduce=True) * y.size(0)

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
        self.is_cifar = False       

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
                #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))
                K.Normalize(torch.FloatTensor((0.5,0.5,0.5)), torch.FloatTensor((0.5,0.5,0.5))))

        self.transforms0 = torch.nn.DataParallel(self.transforms0, [torch.cuda.device(0), torch.cuda.device(1)])
        self.transforms1 = torch.nn.DataParallel(self.transforms1, [torch.cuda.device(0), torch.cuda.device(1)])
        self.transforms2 = torch.nn.DataParallel(self.transforms2, [torch.cuda.device(0), torch.cuda.device(1)])
        self.beta = args.beta
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        
        #self.bt_opt = torch.optim.SGD(self.net.parameters(), lr=lr_, momentum=0., weight_decay=5e-4)
        self.bt_opt = torch.optim.SGD(self.net.parameters(), lr=lr_)
        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()
        self.n_outputs = n_outputs
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
        
        self.n_memories = n_outputs * self.n_memories
        if 'mini' in args.data_file:
            self.memx = torch.FloatTensor(self.n_memories, 3, 128, 128)
        elif 'core' in args.data_file:
            self.memx = torch.FloatTensor(self.n_memories, 3, 224, 224)
        else:
            self.memx = torch.FloatTensor(self.n_memories, n_inputs)
        
        self.memy = torch.LongTensor(self.n_memories)
        self.memt = torch.LongTensor(self.n_memories)
        self.mem_feat = torch.FloatTensor(self.n_memories, n_outputs)
        self.mem_feat.fill_(0)
        self.n_seen_so_far = 0
        self.old_idx = torch.LongTensor(self.n_memories)
        self.old_idx.fill_(-1)
        if args.cuda:
            #self.memx = self.memx.cuda()
            self.memy = self.memy.cuda()
            self.memt = self.memt.cuda()
            self.mem_feat = self.mem_feat.cuda()
            self.old_idx = self.old_idx.cuda()
        unique_y = range(n_outputs)
        zeros = [0 for i in unique_y]
        self.unique_y = [0,1]
        self.mem_cnt = 0
        self.bsz = args.batch_size
        
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
        return 0, int(self.n_outputs)

    def forward(self, x, t, return_feat= False):
        if not self.training:
            x = self.transforms0(x).cuda()
        output = self.net(x)    
        return output
    
    def get_t(self):
        return self.memt[:self.mem_cnt]
        
    def memory_sampling(self,t, ssl=False):
        if ssl:
            sz = 32
        else:
            sz = self.sz
        if t is not None:
            valid_indices = (self.get_t() <= t)
            valid_indices = valid_indices.nonzero().squeeze()
            bx, by, bt = self.memx[valid_indices], self.memy[valid_indices], self.mem_feat[valid_indices]
        else:
            bx, by, bt = self.memx[:self.mem_cnt], self.memy[:self.mem_cnt], self.mem_feat[:self.mem_cnt]
        if bx.size(0) < sz:
            return bx.cuda(), by, bt
        else:
            indices = torch.from_numpy(np.random.choice(bx.size(0), sz, replace=False))
            indices= indices.cuda()
            return bx[indices].cuda(), by[indices], bt[indices]
        
    def add_reservoir(self, x, y,  t):
        n_elem = x.size(0)
        place_left = max(0, self.memx.size(0) - self.mem_cnt)
        if place_left:
            offset = min(place_left, n_elem)
            self.memx[self.mem_cnt: self.mem_cnt + offset].data.copy_(x[:offset])
            self.memy[self.mem_cnt: self.mem_cnt + offset].data.copy_(y[:offset])
            self.memt[self.mem_cnt: self.mem_cnt + offset].fill_(t)
            self.old_idx[self.mem_cnt: self.mem_cnt + offset].fill_(0)
            self.mem_cnt += offset
            self.n_seen_so_far += offset
            if offset == x.size(0):
                return
        x, y = x[place_left:], y[place_left:]
        indices = torch.FloatTensor(x.size(0)).to(x.device).uniform_(0, self.n_seen_so_far).long()
        valid_indices = (indices < self.memx.size(0)).long()
        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer   = indices[idx_new_data]
        self.memx[idx_buffer] = x[idx_new_data].cpu().data.clone()
        self.memy[idx_buffer] = y[idx_new_data]
        self.memt[idx_buffer] = t
        self.n_seen_so_far += x.size(0)
        self.old_idx[idx_buffer] = 0

    def save_logit(self):
        idx = (self.old_idx == 0)
        idx = idx.nonzero().squeeze()
        
        with torch.no_grad():
            if idx.size(0) >= 64: 
                chunks = [idx[x:x+64] for x in range(0, idx.size(0),64)]
                for chunk in chunks:
                    xx = self.transforms0(self.memx[chunk]).cuda()
                    feat = self.net.forward(xx)
                    self.mem_feat[chunk] = feat.data.clone()
            else:
                xx = self.transforms0(self.memx[idx]).cuda()
                feat = self.net.forward(xx)
                self.mem_feat[idx] = feat.data.clone()
            
        self.net.train()
        self.old_idx.fill_(1)
        '''
        for j in idx:
            xx = self.transforms0(self.memx[j].unsqueeze(0)).cuda()
            feat = self.net.forward(xx)
            self.mem_feat[j] = feat.unsqueeze(0).detach()
        '''
    def observe(self, x, t, y):
        #t = info[0]
        self.debugger += 1
        if t != self.current_task:
            self.current_task = t
            self.save_logit()

        self.net.train()
        self.add_reservoir(x,y,t)
        bsz = y.data.size(0)
        
        for j in range(self.n_outer):
            weights_before = deepcopy(self.net.state_dict())
            for _ in range(self.inner_steps):
                self.zero_grad()
                if t > 0:
                    xx, yy, target = self.memory_sampling(t, ssl=True)
                    x1, x2 = self.transforms1(xx), self.transforms2(xx)

                    #x_ = torch.cat([x, xx],0)
                    #x1, x2 = self.transforms1(x_), self.transforms2(x_)
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
                xx, yy, target = self.memory_sampling(t)
                xx = self.transforms0(xx).cuda()
                pred = self.net(xx)
                #pred = torch.gather(pred_, 1, mask)
                loss2 += self.bce(pred, yy)
                loss3 = self.reg * kl(pred , target , self.temp)
            loss = loss1 + loss2 + loss3
            loss.backward()
            self.opt.step()
            
        return 0.
