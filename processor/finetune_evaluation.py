import sys
import argparse
import yaml
import math
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight.torchlight import str2bool
from torchlight.torchlight import DictAction
from torchlight.torchlight import import_class

from .processor import Processor

"""
processor/finetune_evaluation.py:86


"""


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class FT_Processor(Processor):
    """
        Processor for Finetune Evaluation.
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)

        self.loss = nn.CrossEntropyLoss()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch'] > np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def show_best(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = 100 * sum(hit_top_k) * 1.0 / len(hit_top_k)
        accuracy = round(accuracy, 5)
        self.current_result = accuracy
        if self.best_result <= accuracy:
            self.best_result = accuracy
        self.io.print_log('\tBest Top{}: {:.2f}%'.format(k, self.best_result))

    def train(self, epoch):
        # self.model.eval()
        self.model.train()

        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label in loader:
            self.global_step += 1
            # get data
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)
            
            if self.arg.stream == 'joint':
                pass
            elif self.arg.stream == 'motion':
                motion = torch.zeros_like(data)

                motion[:, :, :-1, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]

                data = motion
            elif self.arg.stream == 'bone':
                Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                        (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                        (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

                bone = torch.zeros_like(data)

                for v1, v2 in Bone:
                    bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]
                    
                data = bone
            else:
                raise ValueError

            # forward
            output = self.model(None, data)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)

        self.epoch_info['train_mean_loss']= np.mean(loss_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.show_epoch_info()

    def test(self, epoch):
        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []

        for data, label in loader:
            # get data
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)
            
            if self.arg.stream == 'joint':
                pass
            elif self.arg.stream == 'motion':
                motion = torch.zeros_like(data)

                motion[:, :, :-1, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]

                data = motion
            elif self.arg.stream == 'bone':
                Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                        (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                        (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

                bone = torch.zeros_like(data)

                for v1, v2 in Bone:
                    bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]
                    
                data = bone
            else:
                raise ValueError

            # inference
            with torch.no_grad():
                output = self.model(None, data)
            result_frag.append(output.data.cpu().numpy())

            # get loss
            loss = self.loss(output, label)
            loss_value.append(loss.item())
            label_frag.append(label.data.cpu().numpy())

        self.result = np.concatenate(result_frag)
        self.label = np.concatenate(label_frag)

        self.eval_info['eval_mean_loss']= np.mean(loss_value)
        self.show_eval_info()

        # show top-k accuracy 
        for k in self.arg.show_topk:
            self.show_topk(k)
        self.show_best(1)

        self.eval_log_writer(epoch)

    def tsne(self):
        from sklearn.manifold import TSNE
        from tqdm import tqdm
        import matplotlib.pyplot as plt
        import random
        from feeder.ntu_feeder import PKU_class_mapping
        self.model.eval()
        ntu_loader = iter(self.data_loader['train'])
        pku_loader = iter(self.data_loader['test'])
        ntu_fea = []
        ntu_gt = []
        pku_fea = []
        pku_gt = []

        for i in tqdm(range(50)):

            data, label = next(ntu_loader)
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)
            with torch.no_grad():
                feat = self.model(None, data)
            ntu_fea.append(feat)
            ntu_gt.append(label)

            data, label = next(pku_loader)
            data = data.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)
            with torch.no_grad():
                feat = self.model(None, data)
            pku_fea.append(feat)
            pku_gt.append(label)


        tsne = TSNE(n_components=2, init='pca', random_state=0)
        ntu_fea_stacked = np.concatenate([i.cpu().numpy() for i in ntu_fea])
        ntu_gt_stacked = np.concatenate([i.cpu().numpy() for i in ntu_gt])
        pku_fea_stacked = np.concatenate([i.cpu().numpy() for i in pku_fea])
        pku_gt_stacked = np.concatenate([i.cpu().numpy() for i in pku_gt])

        # ntu_fea_stacked = np.reshape(ntu_fea_stacked, (ntu_fea_stacked.shape[0]*ntu_fea_stacked.shape[2], -1))
        # pku_fea_stacked = np.reshape(pku_fea_stacked, (pku_fea_stacked.shape[0]*pku_fea_stacked.shape[2], -1))

        print('\nDraw TSNE...')
        tSNE_result_training = tsne.fit_transform(ntu_fea_stacked)
        tSNE_result_testing = tsne.fit_transform(pku_fea_stacked)

        # plt.scatter(tSNE_result_training[:, 0], tSNE_result_training[:, 1], c='red', s=1.0)
        # plt.scatter(tSNE_result_testing[:, 0], tSNE_result_testing[:, 1], c='blue', s=1.0)
        #
        # plt.axis("tight")
        # plt.show()

        patial_actions = random.sample(list(np.unique(ntu_gt_stacked)[1:]), 20)  # select 20 randomactions for visualization
        patial_actions.sort()
        fig, ax = plt.subplots(dpi=72 * 9)
        for ind, l in enumerate(patial_actions):
            scatter = ax.scatter(tSNE_result_training[np.where(ntu_gt_stacked == l), 0],
                                 tSNE_result_training[np.where(ntu_gt_stacked == l), 1],
                                 color=plt.cm.tab20(ind), marker='x', linewidth=0.5, alpha=0.9,
                                 s=20)
            scatter = ax.scatter(tSNE_result_testing[np.where(pku_gt_stacked == l), 0],
                                 tSNE_result_testing[np.where(pku_gt_stacked == l), 1],
                                 color=plt.cm.tab20(ind), marker='.', alpha=0.9, s=2, label=PKU_class_mapping[l]) #'action %s' % l)
        # ax.set_xlim(-100.0, 150.0)
        legend1 = ax.legend(*scatter.legend_elements(prop="colors"), loc="lower left", title="Classes")
        ax.legend(markerscale=2, fontsize='small')
        plt.show()



    @staticmethod
    def get_parser(add_help=False):
        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--stream', type=str, default='joint', help='the stream of input')
        parser.add_argument('--mining_epoch', type=int, default=1e6,
                            help='the starting epoch of nearest neighbor mining')
        parser.add_argument('--topk', type=int, default=1, help='topk samples in nearest neighbor mining')

        return parser
