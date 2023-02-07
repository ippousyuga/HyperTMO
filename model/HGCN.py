import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class HGCN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGCN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGCN_conv(in_ch, n_hid[0])
        self.hgc2 = HGCN_conv(n_hid[0], n_hid[1])
        self.hgc3 = HGCN_conv(n_hid[1], n_hid[2])
        self.clf = nn.Linear(n_hid[2], n_class)
        self.fc = nn.Softplus()


    def forward(self, x, G):
        x = self.hgc1(x, G)   
        x = F.leaky_relu(x, 0.25) 
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc2(x, G)
        x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc3(x, G)
        x = F.leaky_relu(x, 0.25)
        x = self.clf(x)
        x = self.fc(x)

        return x



class HGCN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGCN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        x = G.matmul(x)
        if self.bias is not None:
            x = x + self.bias
        return x

