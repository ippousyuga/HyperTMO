import torch
import torch.nn as nn
import torch.nn.functional as F
from model.HGCN import HGCN



def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1   
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return (A + B)


class TMO(nn.Module):
    def __init__(self,in_ch,  classes, omics, HGCN_dims, lambda_epochs=1):
        """
        :param classes: Number of classes
        :param omics: Number of omics data type
        :param HGCN_dims: Dimension of the HGCN
        """
        super(TMO, self).__init__()
        self.omics = omics
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        self.HGCNs = nn.ModuleList([HGCN(in_ch[i], self.classes, HGCN_dims) for i in range(self.omics)])

    def DS_Combin(self, alpha):
        """
        :param alpha: Dirichlet distribution parameters of each omics data type
        :return: Multi-omics integrated dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of omics data type 1
            :param alpha2: Dirichlet distribution parameters of omics data type 2
            :return: integrated dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            p, S, F, u = dict(), dict(), dict(), dict()
            for o in range(2):
                S[o] = torch.sum(alpha[o], dim=1, keepdim=True)
                F[o] = alpha[o]-1
                p[o] = F[o]/(S[o].expand(F[o].shape))
                u[o] = self.classes/S[o]

            pp = torch.bmm(p[0].view(-1, self.classes, 1), p[1].view(-1, 1, self.classes))
            uv1_expand = u[1].expand(p[0].shape)
            pu = torch.mul(p[0], uv1_expand)
            uv_expand = u[0].expand(p[0].shape)
            up = torch.mul(p[1], uv_expand)
            pp_sum = torch.sum(pp, dim=(1, 2), out=None)
            pp_diag = torch.diagonal(pp, dim1=-2, dim2=-1).sum(-1)
            C = pp_sum - pp_diag
            p_a = (torch.mul(p[0], p[1]) + pu + up)/((1-C).view(-1, 1).expand(p[0].shape))
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))
            S_a = self.classes / u_a
            f_a = torch.mul(p_a, S_a.expand(p_a.shape))
            alpha_a = f_a + 1
            return alpha_a

        for o in range(len(alpha)-1):
            if o==0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[o+1])
        return alpha_a

    def forward(self, X, G, y, global_step, idx):   # y:label
        evidence = self.infer(X, G)
        loss = 0
        alpha = dict()
        for o_num in range(len(X)):
            alpha[o_num] = evidence[o_num] + 1
            loss += ce_loss(y[idx], alpha[o_num][idx], self.classes, global_step, self.lambda_epochs)
        alpha_a = self.DS_Combin(alpha)
        evidence_a = alpha_a - 1
        loss += ce_loss(y[idx], alpha_a[idx], self.classes, global_step, self.lambda_epochs)
        loss = torch.mean(loss)
        return evidence_a, loss

    def infer(self, input, input_G):
        """
        :param input: Multi-omics data
        :return: Evidence of every omics data type
        """
        evidence = dict()
        for o_num in range(self.omics):
            evidence[o_num] = self.HGCNs[o_num](input[o_num], input_G[o_num])
        return evidence
