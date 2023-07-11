import random
import torch
import torch.nn as nn

from collections import OrderedDict
from multiprocessing import pool
from src.clients.FederatedClientRolex import FederatedClientRolex
from src.servers.FederatedServer import FederatedServer
from src.utils import ParameterDict

class FederatedServerRolex(FederatedServer):
    def __init__(self
               , model          : nn.Module
               , clients        : list[FederatedClientRolex]
               , rounds         : int
               , server_lr      : float
               , client_fraction: float
               , num_threads    : int
               , post_round_callback  = None
               , device         : str = "cpu"):
        """
            Federated averaging rolex server class
        Args:
            model (nn.Module): pytorch module that clients use in training
            clients (list[FederatedClientRolex]): a list of FedRolex clients
            rounds (int): number of training rounds
            server_lr (float): server learning rate
            client_fraction (float): fraction of random clients used for training per round
            num_threads (int): max number of threads
            post_round_callback (_type_, optional): callable that is called after each round. Defaults to None.
            device (str, optional): "cpu" or "cuda", pytorch device. Defaults to "cpu".
        """
        super().__init__(model, clients, rounds, server_lr, client_fraction, num_threads, post_round_callback, device)

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
                p.map(lambda client: client.fit(self.model, round), selected_clients)
                
            # 2. Calculate total number of traning samples used
            total_samples = sum([len(c) for c in selected_clients])
           
            with torch.no_grad():
                # 3. Average client weights weighted by dataset size, 
                delta_weights_avg    = ParameterDict(self.model, zero=True)
                global_model_weights = ParameterDict(self.model)    
                            
                #weight_update_sum = ParameterDict(self.model, zero=True)
                    
                for client in selected_clients:
                    delta_weights_avg += client.delta_weights / m  #/ N
                #    weight_update_sum += client.masks / N
                #            
                #for key, val in weight_update_sum.items():
                #    weight_update_sum[key] = torch.where(val > 0, val, 1)
                #           
                #delta_weights_avg /= weight_update_sum 

                # 4. Update server weights
                global_model_weights += self.lr * delta_weights_avg

            if self.device == "cuda":
                torch.cuda.empty_cache()
                
            if self.post_round_callback != None:
                self.post_round_callback(self)