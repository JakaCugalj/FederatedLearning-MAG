import torch.nn as nn
from src.clients.FederatedClient import FederatedClient

class FederatedServer():
    def __init__(self
               , model          : nn.Module            
               , clients        : list[FederatedClient]
               , rounds         : int                  
               , learning_rate  : float                
               , client_fraction: float                
               , num_threads    : int                  
               , post_round_callback  = None           
               , device         : str = "cpu"):
        """
            Federated Server base class
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
        self.model               = model
        self.clients             = clients
        self.training_rounds     = rounds
        self.lr                  = learning_rate
        self.client_fraction     = client_fraction
        self.num_threads         = num_threads
        self.post_round_callback = post_round_callback
        self.device              = device
        self.stop_requested      = False
        
    def fit(self):
        pass