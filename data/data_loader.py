import os
from tkinter.messagebox import RETRY

import pandas as pd
import torch



def load_ft(data_folder, omics_list, dataDir):
    """
    load multi-omics data
    """
    data_folder = os.path.join(data_folder, dataDir) 
    label=pd.read_csv(data_folder + '/labels.csv', header = None)
    label_item = torch.LongTensor(label.values)
    cuda = True if torch.cuda.is_available() else False

    data_ft_list = []
    for i in range(len(omics_list)):
        data_ft_list.append((pd.read_csv(os.path.join(data_folder, omics_list[i] + ".csv")).values))

    data_tensor_list = []
    for i in range(len(data_ft_list)):
        data_tensor_list.append(torch.FloatTensor(data_ft_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    if cuda:
        label_item = label_item.cuda()
    return data_tensor_list, label_item.reshape(-1)