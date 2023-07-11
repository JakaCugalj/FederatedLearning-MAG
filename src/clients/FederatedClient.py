import math
import torch.nn as nn

from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from src.utils import EpochMetrics, ParameterDict

class FederatedClient():
    def __init__(self
               , id               : str   
               , training_dataset : Subset
               , epochs           : int   
               , batch_size       : float   
               , lr               : float 
               , momentum         : float 
               , device           : str = "cpu"): 
        """
            Federated Client base class
        Args:
            id (str): client id string
            training_dataset (Subset): client dataset subset
            epochs (int): client learning epochs
            batch_size (int): client minibatch size
            lr (float): client SGD learning rate
            momentum (float): client SGD learning momentum
            device (str, optional): "cpu" or "cuda", pytorch device. Defaults to "cpu".
        """
        self.id                  = id
        self.local_training_data = training_dataset
        self.epochs              = epochs
        self.batch_size          = batch_size
        self.lr                  = lr
        self.momentum            = momentum
        self.device              = device
        self.stop_requested      = False
        self.epoch_metrics       = EpochMetrics()
        self.training_loader     = DataLoader(dataset    = self.local_training_data,
                                              batch_size = int(len(self.local_training_data) * self.batch_size), #int(self.batch_size),
                                              shuffle    = True)
        self.model         : nn.Module     = None
        self.delta_weights : ParameterDict = None
        
        print(f'{self.id}, data_length={len(self.local_training_data)}, batch_size={int(len(self.local_training_data) * self.batch_size)}, num_steps={math.ceil(1 / self.batch_size)}') 
        #self.epochs * len(self.local_training_data) / self.batch_size

    def __len__(self):
        return len(self.local_training_data)

    def fit(self, global_model : nn.Module):
        pass