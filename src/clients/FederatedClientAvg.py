import copy
import dearpygui.dearpygui as dpg

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data.dataset import Subset
from src.clients.FederatedClient import FederatedClient, ParameterDict

class FederatedClientAvg(FederatedClient):
    def __init__(self
               , id               : str   
               , training_dataset : Subset
               , epochs           : int   
               , batch_size       : float   
               , lr               : float 
               , momentum         : float 
               , device           : str = "cpu"
               , parent_tag       : str = None): 
        """
            Federated averaging client. Performs SGD on local data
        Args:
            id (str): client id string
            training_dataset (Subset): client training subset
            epochs (int): client learning epochs
            batch_size (int): client minibatch size
            lr (float): client SGD learning rate
            momentum (float): client SGD learning momentum
            device (str, optional): "cpu" or "gpu", pytorch device. Defaults to "cpu".
            parent_tag (str, optional): User interface dearpygui table tag. Defaults to None.
        """
        
        super().__init__(id, training_dataset, epochs, batch_size, lr, momentum, device)
        
        self.parent_tag = parent_tag
 
        if self.parent_tag != None:
            def __open_parameters_window(sender, app_data, user_data):
                w = dpg.add_window(label=self.id, width=400, height=160, pos=[250, 250])
                self.tag_epochs     = dpg.add_input_int(label="E  (num. of epochs)"           , default_value=self.epochs    , width=100, parent=w)
                self.tag_batch_size = dpg.add_input_int(label="B  (batch size)"               , default_value=self.batch_size, width=100, parent=w)
                self.tag_lr         = dpg.add_input_float(label="LR (learning rate)"          , default_value=self.lr        , width=100, parent=w)
                self.tag_momentum   = dpg.add_input_float(label="Momentum (learning momentum)", default_value=self.momentum  , width=100, parent=w)
                def apply_ui_parameters():
                    self.epochs     = dpg.get_value(self.tag_epochs)
                    self.batch_size = dpg.get_value(self.tag_batch_size)
                    self.lr         = dpg.get_value(self.tag_lr)
                    self.momentum   = dpg.get_value(self.tag_momentum)
                dpg.add_button(label="Apply", callback=apply_ui_parameters, parent=w)
                
            self.tag_table_row = dpg.add_table_row(parent=self.parent_tag)
            with dpg.table_cell(parent=self.tag_table_row):
                dpg.add_button(label=self.id, width=100, callback=__open_parameters_window)
            with dpg.table_cell(parent=self.tag_table_row):
                self.tag_progress_bar = dpg.add_progress_bar(width=-1)
            with dpg.table_cell(parent=self.tag_table_row):
                self.tag_accuracy     = dpg.add_slider_double(label="", no_input=True, min_value=0, max_value=100.0, width=-1)
            with dpg.table_cell(parent=self.tag_table_row):
                self.tag_loss         = dpg.add_slider_double(label="", no_input=True, min_value=0, max_value=3.0, width=-1)
            with dpg.table_cell(parent=self.tag_table_row):
                self.tag_total_epoch_loss = dpg.add_text(default_value="0")
                
    def fit(self, global_model : nn.Module): 
        if self.device == "cuda":
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                self.__fit(global_model)
        else:
            self.__fit(global_model) 
    
    def __fit(self, global_model : nn.Module):
        self.model = copy.deepcopy(global_model)
        
        loss_fn   = nn.CrossEntropyLoss()  
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)     

        self.epoch_metrics.reset() 
         
        for _ in range(self.epochs):
            self.model.train()
   
            for batch_idx, (batch_inputs, batch_labels) in enumerate(self.training_loader):
                if self.stop_requested: break
                
                batch_inputs = batch_inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(batch_inputs)

                loss = loss_fn(outputs, batch_labels)
                loss.backward()
                
                self.epoch_metrics.update(outputs, batch_labels, loss)

                optimizer.step()
                
                if self.parent_tag != None and (batch_idx % 20 == 0 or batch_idx == len(self.training_loader) - 1):
                    dpg.set_value(self.tag_progress_bar, batch_idx / max(len(self.training_loader) - 1, 1))
                    dpg.set_value(self.tag_accuracy, self.epoch_metrics.accuracy * 100)
                    dpg.set_value(self.tag_loss, self.epoch_metrics.loss)
                    dpg.set_value(self.tag_total_epoch_loss, f'{self.epoch_metrics.total_loss:.2f}')
        
        with torch.no_grad(): 
            self.delta_weights = ParameterDict(self.model) - ParameterDict(global_model)