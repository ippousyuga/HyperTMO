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
        x = self.hgc1(x, G)    # 隐藏层1卷积操作
        x = F.leaky_relu(x, 0.25)   # 非线性函数：LeakyReLU(x)=max(0,x)+negative_slope∗min(0,x)
        # torch.nn.functional.leaky_relu(input, negative_slope=0.01, inplace=False) → Tensor
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc2(x, G)    # 隐藏层2卷积操作
        x = F.leaky_relu(x, 0.25)   # 非线性函数：LeakyReLU(x)=max(0,x)+negative_slope∗min(0,x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc3(x, G)    # 隐藏层3卷积操作
        x = F.leaky_relu(x, 0.25)   # 非线性函数：LeakyReLU(x)=max(0,x)+negative_slope∗min(0,x)
        x = self.clf(x)
        x = self.fc(x)

        return x



class HGCN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGCN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))    # 根据input特征和output特征随机构造权重
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))     # if 设置偏置为True则对bias赋值；
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)  # +-区间stdv均匀分布
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        x = G.matmul(x)
        if self.bias is not None:
            x = x + self.bias
        return x

