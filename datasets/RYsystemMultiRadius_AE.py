import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch_geometric.utils import dropout_adj
from datasets.Generator import Radius_attr
from datasets.AuxFunction import FFT
from torch_geometric.data import Data
import pickle
# --------------------------------获取数据-----------------------------
signal_size = 1024
root = "E:\Data\RY_1500"


fault_name = ['正常状态','齿面磨损0.01mm','齿面磨损0.05mm','端面磨损0.08mm','端面磨损0.1mm',
              '从动齿断齿','主动齿断齿','混合断齿']

# label
label = [0,1,1,1,1,1,1,1]



# generate Training Dataset and Testing Dataset
def get_files(sample_length, root, InputType, task,test=False):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    normalname:List of normal data
    dataname:List of failure data
    '''
    # data = []
    Nordata = []
    Abnordata = []
    for i in tqdm(range(len(fault_name))):
        sub_root = os.path.join('/tmp', root, fault_name[i])
        file_name = os.listdir(sub_root)
        fl2 = pd.DataFrame([])
        file_order = [0,1,2,4,5]

        for j in file_order:
            data_name = file_name[j]
            path2 = os.path.join('/tmp', sub_root, data_name)
            fl = pd.read_csv(path2, skiprows=range(15), header=None)[:1024*1000]
            fl = fl.values
            fl = (fl - fl.min()) / (fl.max() - fl.min())  # 数据归一化处理
            fl1 = pd.DataFrame(fl)
            fl2 = pd.concat([fl1, fl2],axis=1)

        node_edge, w = Radius_attr(fl2.values.T)

        data1 = data_load(sample_length, data=fl2, edge_index=node_edge, edge_attr=w, label=label[i], InputType=InputType, task=task)

        if i == 0:
            Nordata, Abnordata = train_test_split(data1, test_size=0.20, random_state=40)
        else:
            _, data1 = train_test_split(data1, test_size=0.1, random_state=40)
            Abnordata += data1

    return Nordata, Abnordata


def data_load(signal_size, data, edge_index,  edge_attr, label, InputType, task):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    '''
    fl = data.values
    graphset= []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        if InputType == "TD":
            x = fl[start:end]
        elif InputType == "FD":
            x = fl[start:end]
            x = FFT(x)
        else:
            print("The InputType is wrong!!")

        subgraph = GenSubgraph(x,edge_index,edge_attr,label)

        graphset.append(subgraph)
        start += signal_size
        end += signal_size

    return graphset

def GenSubgraph(data,edge_index,edge_attr,label):
    x = data.T
    node_features = torch.tensor(x, dtype=torch.float)
    graph_label = torch.tensor([label], dtype=torch.long)  # 获得图标签
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    data = Data(x=node_features, y=graph_label, edge_index=edge_index, edge_attr=edge_attr)
    return data


def train_test_split_order(data_pd, test_size=0.8, num_classes=10):
    train_pd = pd.DataFrame(columns=('data', 'label'))
    val_pd = pd.DataFrame(columns=('data', 'label'))
    for i in range(num_classes):
        data_pd_tmp = data_pd[data_pd['label'] == i].reset_index(drop=True)
        train_pd = train_pd.append(data_pd_tmp.loc[:int((1-test_size)*data_pd_tmp.shape[0]), ['data', 'label']], ignore_index=True)
        val_pd = val_pd.append(data_pd_tmp.loc[int((1-test_size)*data_pd_tmp.shape[0]):, ['data', 'label']], ignore_index=True)
    return train_pd,val_pd


class RYsystemMultiRadius_AE(object):
    num_classes = 10


    def __init__(self, sample_length, data_dir,InputType,task):
        self.sample_length = sample_length
        self.data_dir = data_dir
        self.InputType = InputType
        self.task = task

    def edge_delete(self, data):

        for i in data:
            edge_index = i.edge_index
            new_index, edge_atrr = dropout_adj(edge_index, p=0.1)
            i.edge_index = new_index

        return data

    def data_preprare(self, test=False):
        if len(os.path.basename(self.data_dir).split('.')) == 2:
            with open(self.data_dir, 'rb') as fo:
                list_data = pickle.load(fo, encoding='bytes')
                nor_data, abnor_data = list_data[0], list_data[1]
        else:
            nor_data, abnor_data = get_files(self.sample_length, self.data_dir, self.InputType, self.task, test)
            with open(os.path.join(self.data_dir, "RYsystemMultiRadius_AE.pkl"), 'wb') as fo:
                pickle.dump([nor_data, abnor_data], fo)

        if test:

            # train_dataset, val_dataset = train_test_split(list_data, test_size=0.20, random_state=40)
            dataset = list_data
            return dataset[-20:]

        else:

            train_dataset, val_dataset = nor_data, abnor_data
            return train_dataset, val_dataset
