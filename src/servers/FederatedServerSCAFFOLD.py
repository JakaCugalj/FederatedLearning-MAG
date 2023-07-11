import torch
import torch.nn as nn
import random

from multiprocessing import pool
from src.servers.FederatedServer import FederatedServer
from src.clients.FederatedClientSCAFFOLD import FederatedClientSCAFFOLD, ParameterDict

class FederatedServerSCAFFOLD(FederatedServer):
    def __init__(self
               , model          : nn.Module
               , clients        : list[FederatedClientSCAFFOLD]
               , rounds         : int
               , server_lr      : float
               , client_fraction: float
               , num_threads    : int
               , post_round_callback  = None
               , device         : str = "cpu"):
        """
            SCAFFOLD server class
        Args:
            model (nn.Module): pytorch module that clients use in training
            clients (list[FederatedClientSCAFFOLD]): a list of SCAFFOLD clients
            rounds (int): number of training rounds
            server_lr (float): server learning rate
            client_fraction (float): fraction of random clients used for training per round
            num_threads (int): max number of threads
            post_round_callback (_type_, optional): callable that is called after each round. Defaults to None.
            device (str, optional): "cpu" or "cuda", pytorch device. Defaults to "cpu".
        """    
        super().__init__(model, clients, rounds, server_lr, client_fraction, num_threads, post_round_callback, device)

        self.control = ParameterDict(model, zero=True, device=device) 

    def fit(self):
        self.model.to(self.device)
        
        for c in self.clients:
            c.stop_requested = False
            
        for round in range(self.training_rounds):
            if self.stop_requested: break
            
            N = len(self.clients)
            m = max(int(N * self.client_fraction), 1)
    
            # 0. Sample m clients
            selected_clients_indices = random.sample(range(N), m)    
            selected_clients         = [self.clients[i] for i in selected_clients_indices]

            # 1. LocalSGD -> calculate delta_weights
            with pool.ThreadPool(processes=self.num_threads) as p:
                p.map(lambda client: client.fit(self.model, self.control), selected_clients)

            # 2. Calculate total number of traning samples used
            total_samples = sum([len(c) for c in selected_clients])

            with torch.no_grad():
                # 3. Average client weights and controls weighted by dataset size
                delta_weights_avg    = ParameterDict(self.model, zero=True)
                delta_control_avg    = ParameterDict(self.model, zero=True)
                global_model_weights = ParameterDict(self.model)
        
                for client in selected_clients:
                    delta_weights_avg += client.delta_weights / m #* (len(client) / total_samples)
                    delta_control_avg += client.delta_control / m #* (len(client) / total_samples)
                
                global_model_weights += self.lr * delta_weights_avg
                self.control         += (m / N) * delta_control_avg

            if self.device == "cuda":
                torch.cuda.empty_cache()
                
            if self.post_round_callback != None:
                self.post_round_callback(self)
                

#import torchvision
#import torchvision.transforms as transforms
#
#import torch
#import torch.nn as nn
#
#from FederatedClientSCAFFOLD import FederatedClientSCAFFOLD
#
#from models import GarmentClassifier, CNNNet, CNN
#from utils import dataset_split
#
#transform                = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#local_training_data = torchvision.datasets.MNIST('./data', train=True, transform=transform, download=False)
#local_test_data     = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=False)  
#model               = CNNNet()
#        
#datasets = dataset_split(local_training_data, 10, classes_per_client=2, iid=False, plot=True)
#
#clients = []
#for k in range(10):
#    client = FederatedClientSCAFFOLD(f'client_{k}', datasets[k],
#                                1, 64, 0.001,
#                                0.9, "cpu")
#    clients.append(client)
#
#training_algo = SCAFFOLD(model, clients, 5, 1, 1, 1)
#training_algo.fit()