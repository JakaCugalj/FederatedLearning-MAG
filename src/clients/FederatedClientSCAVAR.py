import copy
import dearpygui.dearpygui as dpg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune

from torch.utils.data.dataset import Subset
from src.clients.FederatedClient import FederatedClient, ParameterDict
from src.clients.FederatedClientRolex import rolling_window_tensor_mask
                    
class FederatedClientSCAVAR(FederatedClient):
    def __init__(self
               , id               : str   
               , training_dataset : Subset
               , model_fraction   : float 
               , epochs           : int   
               , batch_size       : float   
               , lr               : float 
               , momentum         : float 
               , mu               : float 
               , device           : str = "cpu"
               , parent_tag       : str = None): 
        """
            Federated averaging client Rolex. Performs SGD on local data
        Args:
            id (str): client id string
            training_dataset (Subset): client training subset
            model_fraction (float): fraction of the layer model uses at training each round
            epochs (int): client learning epochs
            batch_size (int): client minibatch size
            lr (float): client SGD learning rate
            momentum (float): client SGD learning momentum
            device (str, optional): "cpu" or "gpu", pytorch device. Defaults to "cpu".
            parent_tag (str, optional): User interface dearpygui table tag. Defaults to None.
        """
        
        super().__init__(id, training_dataset, epochs, batch_size, lr, momentum, device)
        
        self.parent_tag     = parent_tag
        self.model_fraction = model_fraction
        self.mu             = mu
        self.control        = ParameterDict()
        
        if self.parent_tag != None:
            def __open_parameters_window(sender, app_data, user_data):
                w = dpg.add_window(label=self.id, width=400, height=180, pos=[250, 250])
                self.tag_epochs     = dpg.add_input_int(label="E  (num. of epochs)"           , default_value=self.epochs        , width=100, parent=w)
                self.tag_batch_size = dpg.add_input_int(label="B  (batch size)"               , default_value=self.batch_size    , width=100, parent=w)
                self.tag_lr         = dpg.add_input_float(label="LR (learning rate)"          , default_value=self.lr            , width=100, parent=w)
                self.tag_momentum   = dpg.add_input_float(label="Momentum (learning momentum)", default_value=self.momentum      , width=100, parent=w)
                self.tag_model_fraction = dpg.add_input_float(label="Model fraction"          , default_value=self.model_fraction, width=100, parent=w)
                def apply_ui_parameters():
                    self.epochs     = dpg.get_value(self.tag_epochs)
                    self.batch_size = dpg.get_value(self.tag_batch_size)
                    self.lr         = dpg.get_value(self.tag_lr)
                    self.momentum   = dpg.get_value(self.tag_momentum)
                    self.model_fraction = dpg.get_value(self.tag_model_fraction)
                    dpg.set_value(self.tag_size, f'{self.model_fraction:.2f}')
                dpg.add_button(label="Apply", callback=apply_ui_parameters, parent=w)
                
            self.tag_table_row = dpg.add_table_row(parent=self.parent_tag)
            with dpg.table_cell(parent=self.tag_table_row):
                dpg.add_button(label=self.id, width=100, callback=__open_parameters_window)
            with dpg.table_cell(parent=self.tag_table_row):
                self.tag_progress_bar = dpg.add_progress_bar(width=-1)
            with dpg.table_cell(parent=self.tag_table_row):
                self.tag_accuracy     = dpg.add_slider_double(label="", no_input=True, min_value=0, max_value=100.0, width=-1)
            with dpg.table_cell(parent=self.tag_table_row):
                self.tag_loss         = dpg.add_slider_double(label="", no_input=True, min_value=0, max_value=3, width=-1)
            with dpg.table_cell(parent=self.tag_table_row):
                self.tag_total_epoch_loss = dpg.add_text(default_value="0") 
            with dpg.table_cell(parent=self.tag_table_row):
                self.tag_size = dpg.add_text(default_value=f'{self.model_fraction:.2f}')

    def fit(self, global_model : nn.Module, global_control : ParameterDict, round : int): 
        if self.device == "cuda":
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                self.__fit(global_model, global_control, round)
        else:
            self.__fit(global_model, global_control, round)
            
    def __fit(self, global_model : nn.Module, global_control : ParameterDict, round : int):    
        self.model = copy.deepcopy(global_model)

        self.masks = ParameterDict({name : rolling_window_tensor_mask(param, round, self.model_fraction)
                                    for name, param in self.model.named_parameters()}, device=self.device)
      
        loss_fn   = nn.CrossEntropyLoss()  
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)    
        
        if self.control.keys() != global_control.keys():
            self.control = ParameterDict(global_control, zero=True)
        
        for name, param in self.model.named_parameters():
            param.data.mul_(self.masks[name])
      
        self.epoch_metrics.reset()    
        
        for _ in range(self.epochs):
            self.model.train()

            for batch_idx, (batch_inputs, batch_labels) in enumerate(self.training_loader):
                if self.stop_requested: break
                
                batch_inputs = (batch_inputs.to(self.device))
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                                          
                outputs = self.model(batch_inputs)

                proximal_term = 0.0
                if self.mu > 0.0:
                    for w_k, w in zip(self.model.parameters(), global_model.parameters()):
                        proximal_term += (w_k - w).norm(2)
              
                #                FedAvg               +         FedProx
                loss = loss_fn(outputs, batch_labels) + (self.mu / 2) * proximal_term
                loss.backward()
          
                self.epoch_metrics.update(outputs, batch_labels, loss)
      
                for (name, param), c, ci in zip(self.model.named_parameters(), global_control.values(), self.control.values()):
                    if param.grad is None:
                        continue
                    param.grad.data.add_(- ci.data + c.data)
                    param.grad.data.mul_(self.masks[name])     
                    
                optimizer.step()
                
                if self.parent_tag != None and (batch_idx % 20 == 0 or batch_idx == len(self.training_loader) - 1):
                    dpg.set_value(self.tag_progress_bar, batch_idx /  max(len(self.training_loader) - 1, 1))
                    dpg.set_value(self.tag_accuracy, self.epoch_metrics.accuracy * 100)
                    dpg.set_value(self.tag_loss, self.epoch_metrics.loss)
                    dpg.set_value(self.tag_total_epoch_loss, f'{self.epoch_metrics.total_loss:.2f}')
                                                                                
        with torch.no_grad(): 
            coefficient        = 1.0 / (self.epochs * len(self.training_loader) * self.lr)
            cplus              = (self.control - global_control + coefficient * (ParameterDict(global_model) - ParameterDict(self.model))) * self.masks
            self.delta_control = (cplus - self.control) * self.masks
            self.delta_weights = (ParameterDict(self.model) - ParameterDict(global_model)) * self.masks
            self.control       = cplus