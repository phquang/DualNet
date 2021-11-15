#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
# other imports
import numpy as np
import pickle as pkl
import os
import logging
from utils import get_logger, get_temp_logger, logging_per_task
from hashlib import md5
from PIL import Image
import argparse
import importlib
import time
import pdb
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from metrics.metrics import confusion_matrix
from tqdm import tqdm
import uuid
import os 
import datetime
from loader import *

import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(linewidth=np.inf)
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Continuum learning')
    ## data args
    parser.add_argument('--use', type = float, default = 0.5)
    parser.add_argument('--data_path', default='/data5/quang/data/nico/',help='path where data is located')
    parser.add_argument('--samples_per_task', type=int, default=-1,help='training samples per task (all if negative)')
    parser.add_argument('--data_file', default='mini',help='data file')
    parser.add_argument('--n_tasks', type=int, default=17)
    parser.add_argument('--train_csv', type=str, default='data/core50_tr.csv')
    parser.add_argument('--test_csv', type=str, default='data/core50_te.csv')
    parser.add_argument('--valid', action='store_false')
    parser.add_argument('--augmentation', action='store_true')
    ## model args
    parser.add_argument('--pretrained', type = str, default = 'no')
    parser.add_argument('--model', type=str, default='single',
                        help='model to train')
    parser.add_argument('--n_memories', type=int, default=50,
                        help='number of memories per task')
    parser.add_argument('--memory_strength', default=1., type=float,
                        help='memory strength (meaning depends on memory)')
    parser.add_argument('--reg', default = 1., type = float)
    parser.add_argument('--grad_sampling_sz', default = 256, type = int)
    ## training args
    parser.add_argument('--inner_steps', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-1,
                        help='SGD learning rate')
    parser.add_argument('--temperature', type=float, default = 1.0, help='temperature for distilation')
    parser.add_argument('--clip', type=float, default = 0.5, help='clip')
    parser.add_argument('--cuda', type=str, default='yes',
                        help='Use GPU?')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--decay', type=int, default = 5)
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--keep_min', type=str, default='yes')
    parser.add_argument('--emb_dim', type=int, default=64)
    parser.add_argument('--n_val', type=float, default=0.2)
    ## logging args
    parser.add_argument('--save_path', type=str, default='results/',
                        help='save models at the end of training')

    parser.add_argument('--shuffle_tasks', type=str, default='no',
                        help='present tasks in order')
    parser.add_argument('--n_runs', type=int, default=1)
    ## mer
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--replay_batch_size', type=int, default=10)
    parser.add_argument('--batches_per_example', type=float, default=1)
    ## ftml
    parser.add_argument('--adapt', type=str, default='no')
    parser.add_argument('--adapt_lr', type=float, default=0.1)
    parser.add_argument('--n_outer', type=int, default=1)
    args = parser.parse_args()

    args.cuda = True if args.cuda == 'yes' else False
    args.adapt = True if args.adapt == 'yes' else False
    args.pretrained = True if args.pretrained == 'yes' else False
    if int(args.seed) > -1:
        torch.cuda.manual_seed_all(args.seed)
    # fname and stuffs
    uid = uuid.uuid4().hex[:8]
    start_time = time.time()
    fname = args.model + '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fname += '_' + uid
    fname = os.path.join(args.save_path, fname)
    # Create the dataset object
    cl_loader = CLwithDomain(args.train_csv, args.test_csv, args.n_tasks, augmentation = args.augmentation)
    train_loader, val_loader, test_loader = cl_loader.build_benchmark(args)
    #n_outputs, n_domains = cl_loader.meta_data()
    
    # model
    Model = importlib.import_module('model.' + args.model)
    n_outputs = 85 if 'min' in args.train_csv else 50
    current_task = -1
    result_a = []
    result_t = []
    model = None
    wandb= None
    LOG = get_logger(['cls_loss', 'acc'], n_runs=args.n_runs, n_tasks=args.n_tasks)

    for run in range(args.n_runs):
        if model is not None: del model
        model = Model.Net(3072, n_outputs, args.n_tasks, args)
        
        if str(args.model) not in ['ftml']:
            model = model.cuda()
        for task , tr_loader in enumerate(train_loader):
            print('\n--------------------------------------')
            print('Run #{} Task #{} --> Train Classifier'.format(run, task))
            print('--------------------------------------\n')

            model.train()
            #tr_loader = DataLoader(train_dataset, shuffle = True, batch_size=10, num_workers=4)
            for _ in range(args.n_epochs):
                for x, y in tqdm(tr_loader, ncols=69):
                    
                    #model.observe(Variable(x).cuda(), task , Variable(y).cuda())
                    model.observe(x.cuda(), task, y.cuda())
                model.on_epoch_end()
            # eval
            model.eval()
            mode='test'
            for task_t, te_loader in enumerate(test_loader):
                if task_t > task: break
                LOG_temp = get_temp_logger(None, ['cls_loss', 'acc'])
                for data, target in te_loader:
                    data, target = data.cuda(), target.cuda()
                    logits = model(data, task_t)
                    loss = F.cross_entropy(logits, target)
                    pred = logits.argmax(dim=1, keepdim=True)
                    LOG_temp['acc'] += [pred.eq(target.view_as(pred)).sum().item() / pred.size(0)]
                    LOG_temp['cls_loss'] += [loss.item()]
                
                logging_per_task(wandb, LOG, run, mode, 'acc', task, task_t, np.round(np.mean(LOG_temp['acc']),3))
                logging_per_task(wandb, LOG, run, mode, 'cls_loss', task, task_t, np.round(np.mean(LOG_temp['cls_loss']),3))
            print('\n{}:'.format(mode))
            print(LOG[run][mode]['acc'])
        
        for mode in ['test']:
            final_accs = LOG[run][mode]['acc'][:,task]
            logging_per_task(wandb, LOG, run, mode, 'final_acc', task, value=np.round(np.mean(final_accs),3))
            best_acc = np.max(LOG[run][mode]['acc'], 1)
            final_forgets = best_acc - LOG[run][mode]['acc'][:,task]
            logging_per_task(wandb, LOG, run, mode, 'final_forget', task, value=np.round(np.mean(final_forgets[:-1]),3))
            final_la = np.diag(LOG[run][mode]['acc'])
            logging_per_task(wandb, LOG, run, mode, 'final_la', task, value=np.round(np.mean(final_la),3))
            print('\n{}:'.format(mode))
            print('final accuracy: {}'.format(final_accs))
            print('average: {}'.format(LOG[run][mode]['final_acc']))
            print('final forgetting: {}'.format(final_forgets))
            print('average: {}\n'.format(LOG[run][mode]['final_forget']))
            print('final LA: {}\n'.format(final_la))
            print('average: {}\n'.format(LOG[run][mode]['final_la']))
            
    print('--------------------------------------')
    print('--------------------------------------')
    print('FINAL Results')
    print('--------------------------------------')
    print('--------------------------------------')

    with open(fname + '.txt', 'w') as text_file:
        print(args, file=text_file)
    for mode in ['test']:
        final_accs = [LOG[x][mode]['final_acc'] for x in range(args.n_runs)]
        final_acc_avg = np.mean(final_accs)
        final_acc_se = 2*np.std(final_accs) / np.sqrt(args.n_runs)
        final_forgets = [LOG[x][mode]['final_forget'] for x in range(args.n_runs)]
        final_forget_avg =  np.mean(final_forgets)
        final_forget_se = 2*np.std(final_forgets) / np.sqrt(args.n_runs)
        final_la = [LOG[x][mode]['final_la'] for x in range(args.n_runs)]
        final_la_avg =  np.mean(final_la)
        final_la_se = 2*np.std(final_la) / np.sqrt(args.n_runs)
        print('\nFinal {} Accuracy: {:.5f} +/- {:.5f}'.format(mode, final_acc_avg, final_acc_se))
        print('\nFinal {} Forget: {:.5f} +/- {:.5f}'.format(mode, final_forget_avg, final_forget_se))
        print('\nFinal {} LA: {:.5f} +/- {:.5f}'.format(mode, final_la_avg, final_la_se))

        if fname is not None:
            with open(fname + '.txt', "a") as text_file:
                
                print('\nFinal {} Accuracy: {:.5f} +/- {:.4f}'.format(mode, final_acc_avg, final_acc_se), file=text_file)
                print('\nFinal {} Forget: {:.5f} +/- {:.4f}'.format(mode, final_forget_avg, final_forget_se), file=text_file)
                print('\nFinal {} LA: {:.5f} +/- {:.4f}'.format(mode, final_la_avg, final_la_se), file=text_file)

