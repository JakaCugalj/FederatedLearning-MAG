import copy
import torch
import torch.nn as nn
import random
import math

from multiprocessing import pool
from src.servers.FederatedServer import FederatedServer
from src.clients.FederatedClientSCAVAR import FederatedClientSCAVAR, ParameterDict

class FederatedServerSCAVAR(FederatedServer):
    def __init__(self
               , model          : nn.Module
               , clients        : list[FederatedClientSCAVAR]
               , rounds         : int
               , server_lr      : float
               , client_fraction: float
               , num_clusters   : int
               , num_threads    : int
               , post_round_callback  = None
               , device         : str = "cpu"):
        """
            FedSCAVAR server class
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
        
        self.num_clusters    = num_clusters if num_clusters > 0 else 1
        self.cluster_y       = [ParameterDict(model, zero=True, device=device) for _ in range(self.num_clusters)]
        self.cluster_indices = [chunk.tolist() for chunk in torch.randperm(len(clients)).chunk(self.num_clusters)]  
        self.control         = ParameterDict(model, zero=True, device=device)

    def fit(self):
        self.model.to(self.device)
        
        for c in self.clients:
            c.stop_requested = False
             
        for round in range(self.training_rounds):
            if self.stop_requested: break
            
            N = len(self.clients)
            m = max(int(N * self.client_fraction), 1)
            
            selected_clients_indices = random.sample(range(N), m)    
            selected_clients         = [self.clients[i] for i in selected_clients_indices]
            
            with pool.ThreadPool(processes=self.num_threads) as p:
                p.map(lambda client: client.fit(self.model, self.control, round), selected_clients)
                                             
            with torch.no_grad():
                delta_weights_avg    = ParameterDict(self.model, zero=True) 
                delta_control_avg    = ParameterDict(self.model, zero=True)  
                global_model_weights = ParameterDict(self.model)
                         
                # num_clusters == num_clients -> FedVARP + SCAFFOLD, drugače ClusterVARP + SCAFFOLD       
                for cluster_index, (indices, cluster) in enumerate(zip(self.cluster_indices, self.cluster_y)):   
                    cluster_weights_avg = ParameterDict(self.model, zero=True)
                    
                    selected_client_indices_in_cluster = list(set(selected_clients_indices) & set(indices))
                    
                    for i in selected_client_indices_in_cluster:
                        delta_weights_avg   += (self.clients[i].delta_weights - cluster) * self.clients[i].masks / m
                        delta_control_avg   +=  self.clients[i].delta_control / m
                        cluster_weights_avg +=  self.clients[i].delta_weights / len(selected_client_indices_in_cluster)
                              
                    delta_weights_avg += cluster * len(indices) / N     

                    if len(selected_client_indices_in_cluster) > 0:
                        self.cluster_y[cluster_index] = cluster_weights_avg  
                                                
                global_model_weights += self.lr * delta_weights_avg
                self.control         += (m / N) * delta_control_avg
                
            if self.device == "cuda":
                torch.cuda.empty_cache()  
                       
            if self.post_round_callback != None:
                self.post_round_callback(self)


class TestnaImplementacija0(FederatedServer):
    def __init__(self
               , model          : nn.Module
               , clients        : list[FederatedClientSCAVAR]
               , rounds         : int
               , server_lr      : float
               , client_fraction: float
               , num_clusters   : int
               , num_threads    : int
               , post_round_callback  = None
               , device         : str = "cpu"):
        """
            FedRolexVARP server class
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
        
        self.num_clusters    = num_clusters if num_clusters > 0 else 1
        self.cluster_indices = [chunk.tolist() for chunk in torch.randperm(len(clients)).chunk(self.num_clusters)]  

    def fit(self):
        self.model.to(self.device)
        
        for c in self.clients:
            c.stop_requested = False
             
        for round in range(self.training_rounds):
            if self.stop_requested: break
            
            N = len(self.clients)
            m = max(int(N * self.client_fraction), 1)
            
            # 0. Sample fraction of clients
            selected_clients_indices = random.sample(range(N), m)    
            selected_clients         = [self.clients[i] for i in selected_clients_indices]
            
            with pool.ThreadPool(processes=self.num_threads) as p:
                p.map(lambda client: client.fit(self.model, {}, round), selected_clients)
                                              
            with torch.no_grad():
                delta_weights_avg    = ParameterDict(self.model, zero=True)  
                global_model_weights = ParameterDict(self.model)
                          
                #TESTNA IMPLEMENTACIJA 0 
                for cluster_index, indices in enumerate(self.cluster_indices):   
                    selected_client_indices_in_cluster = list(set(selected_clients_indices) & set(indices))
                    
                    cluster = ParameterDict(self.model, zero=True)
                                       
                    for i in selected_client_indices_in_cluster:
                        delta_weights_avg += self.clients[i].delta_weights / (2 * m)
                        
                        cluster += self.clients[i].delta_weights / len(selected_client_indices_in_cluster)
                     
                    if len(selected_client_indices_in_cluster) > 0:
                        delta_weights_avg += len(selected_client_indices_in_cluster) * cluster / (2 * m)
                                                               
                global_model_weights += self.lr * delta_weights_avg
                
            if self.device == "cuda":
                torch.cuda.empty_cache()  
                       
            if self.post_round_callback != None:
                self.post_round_callback(self)             
  

class TestnaImplementacija1(FederatedServer):
    def __init__(self
               , model          : nn.Module
               , clients        : list[FederatedClientSCAVAR]
               , rounds         : int
               , server_lr      : float
               , client_fraction: float
               , num_clusters   : int
               , num_threads    : int
               , post_round_callback  = None
               , device         : str = "cpu"):
        """
            FedRolexVARP server class
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
        
        self.num_clusters    = num_clusters if num_clusters > 0 else 1
        self.cluster_indices = [chunk.tolist() for chunk in torch.randperm(len(clients)).chunk(self.num_clusters)]  
        self.control         = ParameterDict(model, zero=True, device=device)

    def fit(self):
        self.model.to(self.device)
        
        for c in self.clients:
            c.stop_requested = False
             
        for round in range(self.training_rounds):
            if self.stop_requested: break
            
            N = len(self.clients)
            m = max(int(N * self.client_fraction), 1)
            
            # 0. Sample fraction of clients
            selected_clients_indices = random.sample(range(N), m)    
            selected_clients         = [self.clients[i] for i in selected_clients_indices]
            
            with pool.ThreadPool(processes=self.num_threads) as p:
                p.map(lambda client: client.fit(self.model, self.control, round), selected_clients)
                                         
            with torch.no_grad():
                delta_weights_avg    = ParameterDict(self.model, zero=True) 
                delta_control_avg    = ParameterDict(self.model, zero=True)  
                global_model_weights = ParameterDict(self.model)
                                                    
                #TESTNA IMPLEMENTACIJA 1, num_clusters == 1 ali num_clusters == num_clients,
                #je algoritem enak SCAFFOLD, pk = 2 / m
                #naredimo fedavg oz. scaffold in pristejemo obtezno (vsota vseh pk je 1.0) sestete clusterje (dodatni premik)
                for cluster_index, indices in enumerate(self.cluster_indices):   
                    selected_client_indices_in_cluster = list(set(selected_clients_indices) & set(indices))
                    
                    cluster = ParameterDict(self.model, zero=True)
                    control = ParameterDict(self.model, zero=True)
                                    
                    for i in selected_client_indices_in_cluster:
                        delta_weights_avg += self.clients[i].delta_weights / m
                        delta_control_avg += self.clients[i].delta_control / m  
                        cluster += self.clients[i].delta_weights / len(selected_client_indices_in_cluster)
                        control += self.clients[i].delta_control / len(selected_client_indices_in_cluster)
                                         
                    if len(selected_client_indices_in_cluster) > 0:
                        pk = len(selected_client_indices_in_cluster) / m  
                        
                        delta_weights_avg += pk * cluster
                        delta_control_avg += pk * control
                                            
                global_model_weights += self.lr * delta_weights_avg
                self.control         += (m / N) * delta_control_avg
                
            if self.device == "cuda":
                torch.cuda.empty_cache()  
                       
            if self.post_round_callback != None:
                self.post_round_callback(self)
 
              
class TestnaImplementacija2(FederatedServer):
    def __init__(self
               , model          : nn.Module
               , clients        : list[FederatedClientSCAVAR]
               , rounds         : int
               , server_lr      : float
               , client_fraction: float
               , num_clusters   : int
               , num_threads    : int
               , post_round_callback  = None
               , device         : str = "cpu"):
        """
            Testna Implementacija 2 server class
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
        
        self.control = ParameterDict(model, zero=True, device=device)

    def fit(self):
        self.model.to(self.device)
        
        for c in self.clients:
            c.stop_requested = False
             
        for round in range(self.training_rounds):
            if self.stop_requested: break
            
            N = len(self.clients)
            m = max(int(N * self.client_fraction), 1)
            
            delta_weights_avg = ParameterDict(self.model, zero=True)
            
            # 0. Sample fraction of clients
            selected_clients_indices = random.sample(range(N), m)    
            selected_clients         = [self.clients[i] for i in selected_clients_indices]
            
            with pool.ThreadPool(processes=self.num_threads) as p:
                p.map(lambda client: client.fit(self.model, {}, round), selected_clients)
                                 
            model_next_step         = copy.deepcopy(self.model)
            model_next_step_weights = ParameterDict(model_next_step)
            
            for client in selected_clients:
                model_next_step_weights += self.lr * client.delta_weights / m
                delta_weights_avg       += client.delta_weights / (2 * m)
                
            selected_clients_indices = random.sample(range(N), m)    
            selected_clients         = [self.clients[i] for i in selected_clients_indices]
            
            with pool.ThreadPool(processes=self.num_threads) as p:
                p.map(lambda client: client.fit(model_next_step, {}, round), selected_clients)
                
            for client in selected_clients:
                delta_weights_avg += client.delta_weights / (2 * m)
                
            with torch.no_grad(): 
                global_model_weights = ParameterDict(self.model)
                
                global_model_weights += self.lr * delta_weights_avg
                
            if self.device == "cuda":
                torch.cuda.empty_cache()  
                       
            if self.post_round_callback != None:
                self.post_round_callback(self)
                
                
class TestnaImplementacija3(FederatedServer):
    def __init__(self
               , model          : nn.Module
               , clients        : list[FederatedClientSCAVAR]
               , rounds         : int
               , server_lr      : float
               , client_fraction: float
               , num_clusters   : int
               , num_threads    : int
               , post_round_callback  = None
               , device         : str = "cpu"):
        """
            Testna Implementacija 3 server class
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

    def fit(self):
        self.model.to(self.device)
        
        for c in self.clients:
            c.stop_requested = False
                 
        for round in range(self.training_rounds):
            if self.stop_requested: break
            
            N = len(self.clients)
            m = max(int(N * self.client_fraction), 1)
            
            moving_model         = copy.deepcopy(self.model)
            delta_weights_avg    = ParameterDict(self.model, zero=True)
            moving_model_weights = ParameterDict(moving_model) 
            R = 3
            for r in range(R):
                # 0. Sample fraction of clients
                selected_clients_indices = random.sample(range(N), m)    
                selected_clients         = [self.clients[i] for i in selected_clients_indices]
                
                with pool.ThreadPool(processes=self.num_threads) as p:
                    p.map(lambda client: client.fit(moving_model, {}, round), selected_clients)
                            
                for client in selected_clients:
                    delta_weights_avg += client.delta_weights / (R * m)
                    moving_model_weights += self.lr * client.delta_weights / m
                
            global_model_weights = ParameterDict(self.model)    
                    
            with torch.no_grad():          
                global_model_weights += self.lr * delta_weights_avg  
           
            if self.device == "cuda":
                torch.cuda.empty_cache()  
                       
            if self.post_round_callback != None:
                self.post_round_callback(self)


class KernelAvg(FederatedServer):
    def __init__(self
               , model          : nn.Module
               , clients        : list[FederatedClientSCAVAR]
               , rounds         : int
               , server_lr      : float
               , client_fraction: float      
               , num_threads    : int
               , kernel_radius        = 1.0
               , kernel_radius_auto   = False
               , post_round_callback  = None
               , device         : str = "cpu"):
        """
            Testna Implementacija 4 server class
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
        
        self.kernel_radius = kernel_radius
        self.kernel_radius_auto = kernel_radius_auto

    def gaussian_kernel(r, h):
        if r < 0 or r > h: return 0
        return math.exp(-(r ** 2) / (2 * (h ** 2)))
    
    def poly6_kernel(r, h):
        if r < 0 or r > h: return 0
        return (315 / (64 * math.pi * h ** 9)) * (h ** 2 - r ** 2) ** 3
    
    def spiky_kernel(r, h):
        if r < 0 or r > h: return 0
        return (15 / (math.pi * h ** 6)) * (h - abs(r)) ** 3

    def fit(self):        
        self.model.to(self.device)
        
        for c in self.clients:
            c.stop_requested = False
                 
        for round in range(self.training_rounds):
            if self.stop_requested: break
            
            N = len(self.clients)
            m = max(int(N * self.client_fraction), 1)
            
                # 0. Sample fraction of clients
            selected_clients_indices = random.sample(range(N), m)    
            selected_clients         = [self.clients[i] for i in selected_clients_indices]
                
            with pool.ThreadPool(processes=self.num_threads) as p:
                p.map(lambda client: client.fit(self.model), selected_clients)
                                 
            with torch.no_grad():
                # 3. Average client weights weighted by dataset size 
                delta_weights_avg    = ParameterDict(self.model, zero=True)
                global_model_weights = ParameterDict(self.model)
                
                radius = self.kernel_radius
                
                if self.kernel_radius_auto:
                    radius = max([c.delta_weights.euclidean_norm() for c in selected_clients]) * 2#+ min([c.delta_weights.euclidean_norm() for c in selected_clients])
                
                selected_clients_distances = [c.delta_weights.euclidean_norm() for c in selected_clients]
                
                selected_clients_kernel_values = [KernelAvg.gaussian_kernel(c.delta_weights.euclidean_norm(), radius) for c in selected_clients]
                #selected_clients_kernel_values = [KernelAvg.poly6_kernel(n, radius) for n in selected_clients_distances]
                #selected_clients_kernel_values = [KernelAvg.spiky_kernel(c.delta_weights.euclidean_norm(), radius) for c in selected_clients]  
                
                kernel_values_sum = sum(selected_clients_kernel_values) 
                
                print(f"radius={radius}, auto={self.kernel_radius_auto}, kernel_values_sum={kernel_values_sum:.5f}")  
                for client, kernel_value, distance in zip(selected_clients, selected_clients_kernel_values, selected_clients_distances):
                    coef = kernel_value / kernel_values_sum   
                    delta_weights_avg += client.delta_weights * coef
                    print(f'{distance:.5f} -> {kernel_value:.5f} -> {coef:.5f}')
                    
                print("\n")
                                
                ## 4. Update server weights
                global_model_weights += self.lr * delta_weights_avg
           
            if self.device == "cuda":
                torch.cuda.empty_cache()  
                       
            if self.post_round_callback != None:
                self.post_round_callback(self)