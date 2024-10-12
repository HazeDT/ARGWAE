import os
from torch_geometric.utils import dropout_adj
import pickle
# --------------------------------获取数据-----------------------------
signal_size = 1024
root = "E:\Data\RY_1500"


fault_name = ['正常状态','齿面磨损0.01mm','齿面磨损0.05mm','端面磨损0.08mm','端面磨损0.1mm',
              '从动齿断齿','主动齿断齿','混合断齿']

# label
label = [0,1,1,1,1,1,1,1]


class RYsystemMultiRadius_AE(object):
    num_classes = 8


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
            print("NO DATA")

        if test:
            pass

        else:
            train_dataset, val_dataset = nor_data, abnor_data
            print(len(train_dataset),len(val_dataset))
            return train_dataset, val_dataset
