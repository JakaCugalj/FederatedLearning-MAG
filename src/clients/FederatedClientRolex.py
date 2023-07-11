import copy
import dearpygui.dearpygui as dpg

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune

from torch.utils.data.dataset import Subset
from src.clients.FederatedClient import FederatedClient, ParameterDict

def random_tensor_mask(tensor, B):
    return rolling_window_tensor_mask(tensor, -1, B)

def rolling_window_tensor_mask(tensor, round, B):
    size = tensor.size()    
    
    new_shape = (-1, ) if len(size) <= 2 else (size[0] * size[1], *size[2:])
   
    mask = torch.zeros_like(tensor).reshape(new_shape)
    
    if round < 0:
        mask_ind = torch.randperm(len(mask))
    else:
        mask_ind = torch.arange(len(mask))
        
    mask[mask_ind[:int(len(mask) * B)]] = 1

    return mask.roll(round, 0).view_as(tensor)
                
class FederatedClientRolex(FederatedClient):
    def __init__(self
               , id               : str   
               , training_dataset : Subset
               , model_fraction   : float 
               , epochs           : int   
               , batch_size       : float   
               , lr               : float 
               , momentum         : float 
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
                
    def fit(self, global_model : nn.Module, round : int): 
        if self.device == "cuda":
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                self.__fit(global_model, round)
        else:
            self.__fit(global_model, round)
            
    def __fit(self, global_model : nn.Module, round : int):    
        self.model = copy.deepcopy(global_model)
        
        self.masks = ParameterDict({name : rolling_window_tensor_mask(param, round, self.model_fraction)
                                    for name, param in self.model.named_parameters()}, device=self.device)
      
        #self.masks = {}  
        #for name, module in self.model.named_children():
        #    if "dropout" in name:
        #        continue
        #    
        #    #custom_from_mask spremeni weights parameter v atribut ki vrne utezi ki jim applyja masko,
        #    #weights ni vec dostopen v self.models.named_parameters() kot je normalno ampak je zdaj original_weights
        #           
        #    self.masks[f'{name}.bias'] = rolling_window_tensor_mask(module.bias, round, self.model_fraction)
        #    prune.custom_from_mask(module, name='bias', mask=self.masks[f'{name}.bias'])
        #    self.masks[f'{name}.weight'] = rolling_window_tensor_mask(module.weight, round, self.model_fraction)
        #    prune.custom_from_mask(module, name='weight', mask=self.masks[f'{name}.weight'])
           
        loss_fn   = nn.CrossEntropyLoss()  
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)    
           
        self.epoch_metrics.reset() 
         
        #v primeru da ne uporabimo 'prune.custom_from_mask' moramo rocno nastaviti utezi ki jih ne rabimo na 0
        #TODO preveri ce je boljse znotraj for loopa mogoce momentum vseeno spremeni utezi
        for name, param in self.model.named_parameters():
            param.data.mul_(self.masks[name])  
              
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
                
                #utezem ki jih ne rabimo nastavimo gradiente na 0 da med ucenjem ostanejo 0
                #je isto kot da damo masko v forward_pre_hooks
                for name, param in self.model.named_parameters():
                    param.grad.mul_(self.masks[name])
                    
                optimizer.step()
                
                if self.parent_tag != None and (batch_idx % 20 == 0 or batch_idx == len(self.training_loader) - 1):
                    dpg.set_value(self.tag_progress_bar, batch_idx /  max(len(self.training_loader) - 1, 1))
                    dpg.set_value(self.tag_accuracy, self.epoch_metrics.accuracy * 100)
                    dpg.set_value(self.tag_loss, self.epoch_metrics.loss)
                    dpg.set_value(self.tag_total_epoch_loss, f'{self.epoch_metrics.total_loss:.2f}')
        
        #naredi spremembe trajne in odstrani maske iz named_buffers(), named_parameters() je spet normalen
        #for name, module in self.model.named_children():
        #    if "dropout" in name:
        #        continue
        #              
        #    prune.remove(module, 'weight')
        #    prune.remove(module, 'bias')  
                                                                              
        with torch.no_grad():   
            self.delta_weights = (ParameterDict(self.model) - ParameterDict(global_model)) * self.masks

            #self.delta_weights = OrderedDict()
            #for (k, w), wi in zip(global_model.named_parameters(), self.model.parameters()):
            #    self.delta_weights[k] = (w.data - wi.data) * self.masks[k]