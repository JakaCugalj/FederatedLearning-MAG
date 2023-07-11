import random
import copy
import torch
import torch.nn as nn

from typing import Union
from multiprocessing import pool
from src.clients.FederatedClientAvg import FederatedClientAvg, ParameterDict
from src.clients.FederatedClientProx import FederatedClientProx
from src.servers.FederatedServer import FederatedServer

class FederatedServerVARP(FederatedServer):
    def __init__(self
               , model          : nn.Module
               , clients        : Union[list[FederatedClientAvg], list[FederatedClientProx]]
               , rounds         : int
               , server_lr      : float
               , client_fraction: float
               , num_threads    : int
               , post_round_callback  = None
               , device         : str = "cpu"):
        """
            FedVARP server class
        Args:
            model (nn.Module): pytorch module that clients use in training
            clients (list[FederatedClient]): a list of clients
            rounds (int): number of training rounds
            server_lr (float): server learning rate
            client_fraction (float): fraction of random clients used for training per round
            num_threads (int): max number of threads
            post_round_callback (_type_, optional): callable that is called after each round. Defaults to None.
            device (str, optional): "cpu" or "cuda", pytorch device. Defaults to "cpu".
        """
        super().__init__(model, clients, rounds, server_lr, client_fraction, num_threads, post_round_callback, device)
        
        self.clients_y = [ParameterDict(model, zero=True, device=device) for _ in range(len(clients))] 

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
                p.map(lambda client: client.fit(self.model), selected_clients)

            with torch.no_grad():
                delta_weights_avg    = ParameterDict(self.model, zero=True)
                global_model_weights = ParameterDict(self.model) 
                
                for i in selected_clients_indices:
                    delta_weights_avg += (self.clients[i].delta_weights - self.clients_y[i]) / m
                
                for y in self.clients_y:
                    delta_weights_avg += y / N
                
                global_model_weights += self.lr * delta_weights_avg        
                    
                for i in selected_clients_indices:
                    self.clients_y[i] = copy.deepcopy(self.clients[i].delta_weights)
                    
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
            if self.post_round_callback != None:
                self.post_round_callback(self)
                
class FederatedServerClusterVARP(FederatedServer):
    def __init__(self
               , model          : nn.Module
               , clients        : Union[list[FederatedClientAvg], list[FederatedClientProx]]
               , rounds         : int
               , server_lr      : float
               , client_fraction: float
               , num_clusters   : int
               , num_threads    : int
               , post_round_callback  = None
               , device         : str = "cpu"):
        """
            ClusterFedVARP server class
        Args:
            model (nn.Module): pytorch module that clients use in training
            clients (list[FederatedClient]): a list of clients
            rounds (int): number of training rounds
            server_lr (float): server learning rate
            client_fraction (float): fraction of random clients used for training per round
            num_clusters (int): number of clusters
            num_threads (int): max number of threads
            post_round_callback (_type_, optional): function that is called after each round. Defaults to None.
            device (str, optional): "cpu" or "cuda", pytorch device. Defaults to "cpu".
        """
        super().__init__(model, clients, rounds, server_lr, client_fraction, num_threads, post_round_callback, device)
        
        self.num_clusters    = num_clusters if num_clusters > 0 else 1
        self.cluster_y       = [ParameterDict(model, zero=True, device=device) for _ in range(self.num_clusters)] 
        self.cluster_indices = [chunk.tolist() for chunk in torch.randperm(len(clients)).chunk(self.num_clusters)] 
            
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
                p.map(lambda client: client.fit(self.model), selected_clients)

            with torch.no_grad():
                delta_weights_avg    = ParameterDict(self.model, zero=True)
                global_model_weights = ParameterDict(self.model)
                
                for cluster_index, indices in enumerate(self.cluster_indices):   
                    cluster_weights_avg = ParameterDict(self.model, zero=True)
                    
                    selected_client_indices_in_cluster = list(set(selected_clients_indices) & set(indices))
                    
                    for i in selected_client_indices_in_cluster:
                        delta_weights_avg   += (self.clients[i].delta_weights - self.cluster_y[cluster_index]) / m
                        cluster_weights_avg +=  self.clients[i].delta_weights / len(selected_client_indices_in_cluster)
                                
                    delta_weights_avg += len(indices) * self.cluster_y[cluster_index] / N     
                                            
                    if len(selected_client_indices_in_cluster) > 0:                       
                        self.cluster_y[cluster_index] = cluster_weights_avg  

                global_model_weights += self.lr * delta_weights_avg  
                           
                #implementacija po pseudokodi
                # 2. Calculate v with selected cluster delta and combined cluster delta 
                #v = ParameterDict(self.model, zero=True)
                #for i in selected_clients_indices:
                #    for cluster_index, cluster in enumerate(self.cluster_y):
                #        if i in self.cluster_indices[cluster_index]:
                #            for layer in v:
                #                v[layer] += (self.clients[i].delta_weights[layer] - cluster[layer]) / m        
                #                
                #for cluster_index, cluster in enumerate(self.cluster_y):
                #    for layer in v:
                #        v[layer] += (len(self.cluster_indices[cluster_index]) * cluster[layer]) / N
                #
                ## 3. Update server weights  
                #for weights, v_vals in zip(self.model.parameters(), v.values()):
                #    weights += self.server_lr * v_vals  
                #    
                ## 4. Update cluster y
                #for cluster_index, cluster in enumerate(self.cluster_y):   
                #    cluster_new = ParameterDict(self.model, zero=True)
                #    
                #    selected_client_indices_in_cluster = list(set(selected_clients_indices) & set(self.cluster_indices[cluster_index]))
                #    
                #    for i in selected_client_indices_in_cluster:
                #        for layer in cluster_new:
                #            cluster_new[layer] += self.clients[i].delta_weights[layer] / len(selected_client_indices_in_cluster)
                #    
                #    if len(selected_client_indices_in_cluster) > 0:
                #        self.cluster_y[cluster_index] = cluster_new  
                
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
            if self.post_round_callback != None:
                self.post_round_callback(self)