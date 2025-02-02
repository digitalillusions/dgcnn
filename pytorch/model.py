#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from GPUtil import showUtilization

def gpu_usage(message=''):
    print(message)
    showUtilization()
    print('------------------')


def knn(x, k):
    with torch.no_grad():
        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = - xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
        return idx


def knn_iterative(x, k):
    with torch.no_grad():
        ret = []
        for y in x:
            r = 2*torch.mm(y.t(), y)
            diag = r.diag().unsqueeze(0).expand_as(r)
            dist = diag + diag.t() - 2*r
            ret.append(dist.topk(k=k, largest=False, dim=-1)[1])
        return torch.stack(ret)
    

def get_graph_feature(x, k=20, idx=None, local=False):
    batch_size, num_dims, num_points = x.size()
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)

    # idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    device = "cuda:0" if idx.is_cuda else "cpu"
    idx = idx+torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx.view(-1)
 
    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    if local:
        feature = (feature-x).permute(0, 3, 1, 2)
    else:
        feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2)

    return feature


def get_graph_feature_iterative(x, idx):
    k = idx.size()[-1]
    for item, item_idx in zip(x, idx):
        y = item.t()[item_idx]
        yield (y - item.t().unsqueeze(1).expand(-1,k,-1)).permute(2,0,1)


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class Align3D(nn.Module):
    def __init__(self):
        super(Align3D, self).__init__()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(64)

        self.conv1 = nn.Conv2d(3, 64, 1)
        self.conv2 = nn.Conv2d(64, 128, 1)
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 9)

    def forward(self, x, idx):
        batch_size = x.size()[0]
        x = torch.stack([feat for feat in get_graph_feature_iterative(x, idx)])
        x = F.relu(self.bn1(self.conv1(x)))
        x = x.max(dim=-1, keepdim=False)[0]

        x = torch.stack([feat for feat in get_graph_feature_iterative(x, idx)])
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.max(dim=-1, keepdim=False)[0]
        x = x.max(dim=-1, keepdim=False)[0]

        x = F.relu(self.bn3(self.linear1(x)))
        x = self.linear2(x)
        # iden = torch.autograd.Variable(torch.eye(3)).view(1, 9).expand(batch_size, -1)
        # iden = x.new_tensor(torch.eye(3)).view(1, 9).expand(batch_size, -1)
        iden = torch.eye(3, device=x.device, dtype=x.dtype).view(1, 9).expand(batch_size, -1)
        # if x.is_cuda:
        #     iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class SLGCNN(nn.Module):
    """
    Static local graph convolutional neural network. The specific purpose of this module is to learn translation
    and permutation invariant local features that do not depend on the absolute positions of particles. Additionally
    the neighborhoods of the particles remain constant throughout the entire graph. Finally, the neighborhoods are not
    computed using k-nn but rather a distance heuristic which means that padding has to be applied.
    """
    def __init__(self, in_dim=3, out_dim=1, k=20, do_alignment=False):
        super(SLGCNN, self).__init__()

        self.k = k

        self.align = Align3D()
        self.do_alignment = do_alignment

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(128, out_dim, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, idx=None):
        # gpu_usage("Before knn")
        if idx is None:
            idx = knn_iterative(x, self.k)

        # Apply 3d point transformation
        if self.do_alignment:
            trans = self.align(x, idx)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)

        # x = get_graph_feature(x, k=self.k, idx=idx, local=True)
        x = torch.stack([feat for feat in get_graph_feature_iterative(x, idx)])
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        # x = get_graph_feature(x1, k=self.k, idx=idx, local=True)
        x = torch.stack([feat for feat in get_graph_feature_iterative(x1, idx)])
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        # x = get_graph_feature(x2, k=self.k, idx=idx, local=True)
        x = torch.stack([feat for feat in get_graph_feature_iterative(x2, idx)])
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        # x = get_graph_feature(x3, k=self.k, idx=idx, local=True)
        x = torch.stack([feat for feat in get_graph_feature_iterative(x3, idx)])
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return x.transpose(2,1)


class SLGCNNlite(nn.Module):
    """
    Static local graph convolutional neural network. The specific purpose of this module is to learn translation
    and permutation invariant local features that do not depend on the absolute positions of particles. Additionally
    the neighborhoods of the particles remain constant throughout the entire graph. Finally, the neighborhoods are not
    computed using k-nn but rather a distance heuristic which means that padding has to be applied.
    """
    def __init__(self, in_dim=3, out_dim=1, k=20, do_alignment=False):
        super(SLGCNNlite, self).__init__()

        self.k = k

        self.align = Align3D()
        self.do_alignment = do_alignment

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn6 = nn.BatchNorm1d(64)

        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, 32, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(64, out_dim, kernel_size=1, bias=False),
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, idx=None):
        # gpu_usage("Before knn")
        if idx is None:
            idx = knn_iterative(x, self.k)

        # Apply 3d point transformation
        if self.do_alignment:
            trans = self.align(x, idx)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)

        x = torch.stack([feat for feat in get_graph_feature_iterative(x, idx)])
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = torch.stack([feat for feat in get_graph_feature_iterative(x1, idx)])
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = torch.stack([feat for feat in get_graph_feature_iterative(x2, idx)])
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.stack([feat for feat in get_graph_feature_iterative(x3, idx)])
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return x.transpose(2,1)


if __name__ == "__main__":
    x = torch.rand(size=(4, 3, 10000), dtype=torch.float32)
    args = type('dgcnn_args', (), {'k': 10, 'emb_dims': 256, 'dropout': 0.2})

    # pointnet = PointNet(args)
    # y = pointnet(x)
    # print(f"Point net test output: {y.size()}")

    # dgcnn = DGCNN(args)
    # y = dgcnn(x)
    # print(f"DGCNN test output: {y.size()}")

    align = Align3D()
    idx = knn(x, 10)
    y = align(x, idx)
    print(f"Alignment network output: {y.size()}")

    # slgcnn = SLGCNN(in_dim=3, out_dim=1)
    # y = slgcnn(x)
    # print(f"SLGCNN test output: {y.size()}")

    # slgcnn_lite = SLGCNNlite(in_dim=3, out_dim=1)
    # y = slgcnn_lite(x)
    # print(f"SLGCNN lite test output: {y.size()}")

    # if torch.cuda.is_available():
    #     slgcnn.to("cuda:0")
    #     y = slgcnn(x.cuda())
    #     print(f"SLGCNN on GPU output: {y.size()}")

    # Test the iterative knn and iterative feature retrieval
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # neigh = knn(x.to(device=device), 10)
    # neigh_iter = knn_iterative(x.to(device=device), 10)
    # feature = torch.stack([feat for feat in get_graph_feature_iterative(x, neigh_iter)])
    # for i in zip(neigh, neigh_iter):
    #     print(i[0][torch.eq(i[0], i[1]) == 0])
    #     print(i[1][torch.eq(i[0], i[1]) == 0])
