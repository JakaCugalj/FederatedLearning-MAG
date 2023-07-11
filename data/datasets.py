import json
import os
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
import torch


#https://github.com/TalwalkarLab/leaf
#./preprocess.sh -s niid --sf 0.2 -k 0 -t sample -tf 0.8 (small-sized dataset) ('-tf 0.8' reflects the train-test split used in the FedAvg paper)

#https://github.com/TalwalkarLab/leaf/tree/master/data/shakespeare
#https://github.com/wenzhu23333/Federated-Learning/blob/master/utils/dataset.py

ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"

class Shakespeare(Dataset):
    def __init__(self, train=True):
        super(Shakespeare, self).__init__()
        train_clients, train_groups, train_data_temp, test_data_temp = read_data("./data/Shakespeare/train",
                                                                                 "./data/Shakespeare/test")
        self.train = train

        if self.train:
            self.dic_users = {}
            train_data_x = []
            train_data_y = []
            for i in range(len(train_clients)):
                self.dic_users[i] = set()
                l = len(train_data_x)
                cur_x = train_data_temp[train_clients[i]]['x']
                cur_y = train_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    self.dic_users[i].add(j + l)
                    train_data_x.append(cur_x[j])
                    train_data_y.append(cur_y[j])
            self.data    = train_data_x
            self.targets = train_data_y
        else:
            test_data_x = []
            test_data_y = []
            for i in range(len(train_clients)):
                cur_x = test_data_temp[train_clients[i]]['x']
                cur_y = test_data_temp[train_clients[i]]['y']
                for j in range(len(cur_x)):
                    test_data_x.append(cur_x[j])
                    test_data_y.append(cur_y[j])
            self.data    = test_data_x
            self.targets = test_data_y
            
        self.classes = [c for c in ALL_LETTERS]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence, target = self.data[index], self.targets[index]
        indices = torch.LongTensor([ALL_LETTERS.find(c) for c in sentence])
        target  = ALL_LETTERS.find(target)
        return indices, target
    
    def get_client_dic(self):
        if self.train:
            return self.dic_users #dict 138 clientov in njihovi indexov
        else:
            exit("The test dataset do not have dic_users!")

#https://github.com/TalwalkarLab/leaf/blob/master/models/utils/model_utils.py
def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda : None)
    
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data

#https://github.com/TalwalkarLab/leaf/blob/master/models/utils/model_utils.py
def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data