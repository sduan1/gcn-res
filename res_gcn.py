import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa


class ResGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index, edge_weight):
        super(ResGCN, self).__init__()
        skip = True
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.blocks = []



        self.block1 = ResGCNBlock(in_channels, in_channels, edge_index, edge_weight, use_identity=skip)
        self.block2 = ResGCNBlock(in_channels, in_channels, edge_index, edge_weight, use_identity=skip)
        self.block3 = ResGCNBlock(in_channels, in_channels, edge_index, edge_weight, use_identity=skip)
        self.block4 = ResGCNBlock(in_channels, in_channels, edge_index, edge_weight, use_identity=skip)
        self.block5 = ResGCNBlock(in_channels, in_channels, edge_index, edge_weight, use_identity=skip)
        self.block6 = ResGCNBlock(in_channels, in_channels, edge_index, edge_weight, use_identity=skip)
        self.block7 = ResGCNBlock(in_channels, in_channels, edge_index, edge_weight, use_identity=skip)


        self.final = GCNConv(in_channels, out_channels, normalize=True)
        #self.block5 = ResGCNBlock(16, out_channels, edge_index, edge_weight, use_identity=True)
    def forward(self, data):
        x = data.x
        #print('input: {} | norm : {}'.format(x, x.norm()))

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # x = self.block5(x)
        # x = self.block6(x)
        # x = self.block7(x)
        #print('output: {} | norm : {}'.format(x, x.norm()))
        if self.training:
            print('std: {}'.format(torch.std(x.norm(dim=1))))
        x = self.final(x, self.edge_index, self.edge_weight)
        x = F.log_softmax(x, dim=1)
        return x


class ResGCNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_index, edge_weight, use_identity=True, block_size=1, use_gcd=False):
        super(ResGCNBlock, self).__init__()

        self.use_identity = use_identity
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.conv1 = GCNConv(in_channels, in_channels, normalize=True).cuda()
        self.conv2 = GCNConv(in_channels, in_channels, normalize=True).cuda()
        #self.conv3 = GCNConv(in_channels, in_channels, normalize=True).cuda()

    def forward(self, x):
        identity = x
        x = F.relu(self.conv1(x, self.edge_index, self.edge_weight))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, self.edge_index, self.edge_weight))
        x = F.dropout(x, training=self.training)
        #x = F.relu(self.conv3(x, self.edge_index, self.edge_weight))
        x = F.normalize(x)
        if self.use_identity:
            x = (x + identity)/2
        return x
