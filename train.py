import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sched import scheduler
from data.data_loader import load_ft
from model.HGCN import HGCN
from model.TMO import TMO
from utils.hypergraph_utils import gen_trte_inc_mat


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def train_epoch(data_list, g_list, label, model, optimizer, scheduler_dict, epoch, idx_tr=[]):
    """
    :param data_list: The omics features
    :param g_list: The laplace incidence matrix
    :param label: Sample labels
    :param model: The HyperTMO model
    :param optimizer: Training optimizer, Adam optimizer
    :param epoch: Current training epoch
    :param idx_tr: The index of train set
    """
    scheduler_dict.step()
    model.train()
    loss_meter = AverageMeter()
    criterion = torch.nn.CrossEntropyLoss()

    optimizer.zero_grad()
    if len(data_list) >= 2:
        evidence_a, loss = model(data_list, g_list, label, epoch, idx_tr)
    else:
        ci = model(data_list[0], g_list[0])
        loss = torch.mean(criterion(ci[idx_tr], label[idx_tr]))

    loss.backward()
    optimizer.step()
    loss_meter.update(loss.item())



def test_epoch(data_list, label, g_list, te_idx, model, epoch, idx_list_all):
    """
    :param data_list: The omics features 
    :param label: Sample labels
    :param g_list: The laplace incidence matrix
    :param te_idx: The index of test set
    :param model: The HyperTMO model
    :param epoch: Current training epoch
    :param idx_list_all: The index of dataset
    """
    model.eval()
    loss_meter = AverageMeter()
    with torch.no_grad():
        if len(data_list) >= 2:
            evidence_a, loss = model(data_list, g_list, label, epoch, idx_list_all)
            loss_meter.update(loss.item())
        else:
            evidence_a = model(data_list[0], g_list[0])
    c = evidence_a[te_idx, :]
    prob = F.softmax(c, dim=1).data.cpu().numpy()
    return prob


def train_model(data_tensor_list, model, g_list, labels_tensor, criterion, optimizer, scheduler, num_epochs, print_freq, 
                  idx_dict, num_class):
    """
    :param data_tensor_list: The omics features
    :param model: The HGCN model
    :param g_list: The laplace incidence matrix
    :param labels_tensor: Sample labels
    :param optimizer: Training optimizer, Adam optimizer
    :param criterion: Cross-entropy criterion
    :param num_epochs: The epochs
    :param print_freq: Print frequency
    :param idx_dict: The index of train set and test set
    :param num_class: Number of classes
    """
    best_acc = 0.0
    best_f1 = 0.0
    best_auc =0.0
    best_macro = 0.0
    for epoch in range(num_epochs):
        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            idx = idx_dict['tr'] if phase == 'train' else idx_dict['te']
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(data_tensor_list[0], g_list[0])
                _, preds = torch.max(outputs, 1)
                if phase == 'train':
                    loss = torch.mean(criterion(outputs[idx], labels_tensor[idx]))
                    loss.backward()
                    optimizer.step()
                if epoch % 200 == 0:
                    print()
            running_loss += loss.item() * data_tensor_list[0].size(0)
            running_corrects += torch.sum(preds[idx] == labels_tensor.data[idx])
            epoch_loss = running_loss / len(idx)
            epoch_acc = running_corrects.double() / len(idx)
            if epoch % print_freq == 0:
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
            if f1_score(labels_tensor[idx_dict["te"]].cpu(), preds[idx_dict['te']].cpu(), average='weighted') > best_f1:
                best_f1 = f1_score(labels_tensor[idx_dict["te"]].cpu(), preds[idx_dict['te']].cpu(), average='weighted')
            if (f1_score(labels_tensor[idx_dict["te"]].cpu(), preds[idx_dict['te']].cpu(), average='macro') > best_macro and num_class > 2):
                best_macro = f1_score(labels_tensor[idx_dict["te"]].cpu(), preds[idx_dict['te']].cpu(), average='macro')
            if (num_class == 2 and roc_auc_score(labels_tensor[idx_dict["te"]].cpu(), F.softmax(outputs[idx_dict["te"]], dim=1).data.cpu().numpy()[:, 1]) > best_auc  ):
                best_auc = roc_auc_score(labels_tensor[idx_dict["te"]].cpu(), F.softmax(outputs[idx_dict["te"]], dim=1).data.cpu().numpy()[:, 1])
        if epoch % print_freq == 0:
            print(f'Best val Acc: {best_acc:4f}')
            print('-' * 20)
    print(f'Best val Acc: {best_acc:4f} in {best_epoch}')
    print(f'Best val f1: {best_f1:4f}')
    return best_acc.cpu(), best_f1, best_macro, best_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_dir', '-fd', type=str, required=True, help='The dataset file folder.')
    parser.add_argument('--seed', '-s', type=int, default=20, help='Random seed, default=20.')
    parser.add_argument('--num_epoch', '-ne', type=int, default=40000, help='Training epochs, default: 40000.')
    parser.add_argument('--lr_e', '-lr', type=float, default=0.001, help='Learning rate, default: 0.001.')
    parser.add_argument('--dim_he_list', '-dh', nargs = '+', type=int, default=[400, 200, 200], help='Hidden layer dimension of HGCN.')
    parser.add_argument('--num_class', '-nc', type=int, required=True, help='Number of classes.')
    parser.add_argument('--k_neigs', '-kn', type=int, default=4, help='Number of vertices in hyperedge.')
    args = parser.parse_args()

    data_folder = 'data'
    omics_list = ['miRNA','meth','mRNA']
    test_inverval = 50
    num_omics = len(omics_list)
    cuda = True if torch.cuda.is_available() else False
    idx_dict = {}

    data_tensor_list, labels_tensor = load_ft(data_folder, omics_list, args.file_dir)
    dim_list = [x.shape[1] for x in data_tensor_list]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    acc_res, F1_res, AUC_res = [],[],[]
    for idx_train, idx_test in skf.split(pd.DataFrame(data = data_tensor_list[0].cpu()), 
        pd.DataFrame(labels_tensor.cpu())):
        g_list = []
        g = gen_trte_inc_mat(data_tensor_list, args.k_neigs)
        for i in range(len(data_tensor_list)):
            g_list.append(torch.Tensor(g[i]).cuda())
        idx_list_all = list(range(g_list[0].shape[0]))
        dim_list = [x.shape[1] for x in data_tensor_list]
        if num_omics >= 2:
            model_dict = TMO(dim_list, args.num_class, num_omics, args.dim_he_list)
        else:
            model_dict = HGCN(dim_list[0], args.num_class, args.dim_he_list)
        if cuda:
            model_dict.cuda()

        print("\nTraining...")
        optim_dict = torch.optim.Adam(model_dict.parameters(), lr=args.lr_e, weight_decay=0.0005)
        scheduler_dict = torch.optim.lr_scheduler.MultiStepLR(optim_dict, milestones=[100], gamma=0.9)
        best_acc = 0.0
        best_f1 = 0.0
        best_macro = 0.0
        best_auc = 0.0
        idx_dict["tr"] = idx_train
        idx_dict["te"] = idx_test
        if num_omics >= 2:
            for epoch in range(args.num_epoch + 1):
                train_epoch(data_tensor_list, g_list, labels_tensor,
                            model_dict, optim_dict, scheduler_dict, epoch = epoch, idx_tr = idx_dict["tr"])
                te_prob = test_epoch(data_tensor_list, labels_tensor, g_list, idx_dict["te"], 
                    model_dict, epoch, idx_list_all)
                if accuracy_score(labels_tensor[idx_dict["te"]].cpu(), te_prob.argmax(1)) > best_acc:
                    best_acc = accuracy_score(labels_tensor[idx_dict["te"]].cpu(), te_prob.argmax(1))
                if f1_score(labels_tensor[idx_dict["te"]].cpu(), te_prob.argmax(1), average='weighted') > best_f1:
                    best_f1 = f1_score(labels_tensor[idx_dict["te"]].cpu(), te_prob.argmax(1), average='weighted')
                if (f1_score(labels_tensor[idx_dict["te"]].cpu(), te_prob.argmax(1), average='macro') > best_macro
                    and args.num_class > 2):
                    best_macro = f1_score(labels_tensor[idx_dict["te"]].cpu(), te_prob.argmax(1), average='macro')
                if (args.num_class == 2 and roc_auc_score(labels_tensor[idx_dict["te"]].cpu(), te_prob[:, 1]) > best_auc):
                    best_auc = roc_auc_score(labels_tensor[idx_dict["te"]].cpu(), te_prob[:, 1])
                if epoch % test_inverval == 0:
                    print("\nTest: Epoch {:d}".format(epoch))
                    if args.num_class == 2:
                        print("Test ACC: {:.3f}".format(accuracy_score(labels_tensor[idx_dict["te"]].cpu(), 
                            te_prob.argmax(1))))
                        print("Test F1: {:.3f}".format(f1_score(labels_tensor[idx_dict["te"]].cpu(), 
                            te_prob.argmax(1))))
                        print("Test AUC: {:.3f}".format(roc_auc_score(labels_tensor[idx_dict["te"]].cpu(), 
                            te_prob[:, 1])))
                        print("Best Test ACC: {:.3f}".format(best_acc))
                    else:
                        print("Test ACC: {:.3f}".format(accuracy_score(labels_tensor[idx_dict["te"]].cpu(),
                            te_prob.argmax(1))))
                        print("Test F1 weighted: {:.3f}".format(
                            f1_score(labels_tensor[idx_dict["te"]].cpu(), te_prob.argmax(1), average='weighted')))
                        print("Test F1 macro: {:.3f}".format(f1_score(labels_tensor[idx_dict["te"]].cpu(),
                            te_prob.argmax(1), average='macro')))
                        print("Best Test ACC: {:.3f}".format(best_acc))
            if args.num_class == 2:
                F1_res.append(best_f1)
                acc_res.append(best_acc)
                AUC_res.append(best_auc)
            else :
                F1_res.append(best_f1)
                acc_res.append(best_acc)
                AUC_res.append(best_macro)
        else:
            criterion = torch.nn.CrossEntropyLoss()
            best_acc, best_f1, best_macro, best_auc = train_model(model_dict, criterion, optim_dict, scheduler_dict, 
                args.num_epoch, 50,  data_tensor_list, g_list, labels_tensor, idx_dict, args.num_class)
            if args.num_class == 2:
                F1_res.append(best_f1)
                acc_res.append(best_acc)
                AUC_res.append(best_auc)
            else :
                F1_res.append(best_f1)
                acc_res.append(best_acc)
                AUC_res.append(best_macro)
    print('5-fold performance: Acc(%.4f ± %.4f)  F1(%.4f ± %.4f)  AUC/F1_mac(%.4f ± %.4f)'
        % (np.mean(acc_res), np.std(acc_res), np.mean(F1_res), np.std(F1_res)
        , np.mean(AUC_res), np.std(AUC_res)))
    print('Finished!')


