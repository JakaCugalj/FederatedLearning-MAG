from collections import OrderedDict
import math
import dearpygui.dearpygui as dpg

import torchvision
import torchvision.transforms as transforms

import os
import random
import torch
import torch.nn as nn
import numpy as np

import threading

from src.servers.FederatedServerAvg import FederatedServerAvg, FederatedClientAvg, FederatedClientProx
from src.servers.FederatedServerSCAFFOLD import FederatedServerSCAFFOLD, FederatedClientSCAFFOLD
from src.servers.FederatedServerVARP import FederatedServerVARP, FederatedServerClusterVARP
from src.servers.FederatedServerRolex import FederatedServerRolex, FederatedClientRolex
from src.servers.FederatedServerSCAVAR import FederatedServerSCAVAR,FederatedClientSCAVAR
from src.servers.FederatedServerSCAVAR import TestnaImplementacija0, TestnaImplementacija1, TestnaImplementacija2, TestnaImplementacija3, KernelAvg

from src.utils import EpochMetrics
from torch.utils.data import Subset
from src.models import GarmentClassifier, CNN, CIFAR10Net, CharLSTM, LeNet5
from data.datasets import Shakespeare
from src.utils import dataset_split, dataset_split2

random_seed = 12345687

class ServerControlWidget(): 
    def __init__(self, id : str = "ServerControl"):
        self.id      = id
        self.width   = 800
        self.height  = 800     
        self.parameter_section_height = 200
        self.plot_height              = 300  
            
        self.clients      = []
        self.test_metrics = EpochMetrics()
        self.settings     = OrderedDict()

        self.checkpoint_folder = f"./checkpoints/{self.id}" 
        
        self.server = None
        self.model  = None
        self.thread = None
        
        self.tag_dragpoints = []
        self.tag_dragline_acc_start  = None
        self.tag_dragline_acc_end    = None
        self.tag_dragline_loss_start = None
        self.tag_dragline_loss_end   = None

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        #LINE SERIES THEMES
        with dpg.theme() as realtime_line_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (150, 255, 0), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Diamond, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 2, category=dpg.mvThemeCat_Plots)
                
        with dpg.theme() as avg_line_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 255, 52), category=dpg.mvThemeCat_Plots)
                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_None, category=dpg.mvThemeCat_Plots)
        
        
        #LOAD CHECKPOINT FILE DIALOG
        self.tag_checkpoint_dialog = dpg.add_file_dialog(directory_selector = False,
                                                         show     = False,
                                                         callback = self.load_callback,
                                                         width    = 700,
                                                         height   = 400)
        dpg.add_file_extension(".*",
                               parent = self.tag_checkpoint_dialog)
        dpg.add_file_extension(".pt",
                               color       = (0, 255, 0, 255),
                               custom_text = "[Pytorch checkpoint]",
                               parent      = self.tag_checkpoint_dialog)
        
        #SELECT CHECKPOINT FOLDER FILE DIALOG   
        self.tag_checkpoint_folder_dialog = dpg.add_file_dialog(directory_selector = True,
                                                                show     = False,
                                                                callback = self.load_callback,
                                                                width    = 700,
                                                                height   = 400)
        dpg.add_file_extension(".*",
                               parent = self.tag_checkpoint_folder_dialog)
        dpg.add_file_extension(".pt",
                               color       = (0, 255, 0, 255),
                               custom_text = "[Pytorch checkpoint]",
                               parent      = self.tag_checkpoint_folder_dialog)
        
        #MAIN PARAMETERS / PLOTS WINDOW 
        with dpg.window(tag=self.id, label=self.id, width=self.width, height=self.height,
                        pos=[10, 10], menubar=True, collapsed=False):
            #ZGORNJI MENU
            with dpg.menu_bar():
                with dpg.menu(label="Deploy"):
                    dpg.add_menu_item(label = "FedAvg",
                                      callback = self.deploy_fedavg)
                    dpg.add_menu_item(label = "FedProx",
                                      callback = self.deploy_fedprox)
                    dpg.add_menu_item(label = "SCAFFOLD",
                                      callback = self.deploy_scaffold)
                    dpg.add_menu_item(label = "FedVARP",
                                      callback = self.deploy_fedvarp)
                    dpg.add_menu_item(label = "FedRolex",
                                      callback = self.deploy_fedrolex)
                    dpg.add_menu_item(label = "FedSCAVAR",
                                      callback = self.deploy_scavar)   
                    dpg.add_menu_item(label = "FedKernelAvg",
                                      callback = self.deploy_kernelavg,
                                      show=False)   
                    
                dpg.add_menu_item(label = "Train", 
                                  callback = self.fit,
                                  check    = True)
                dpg.add_menu_item(label = "Test", 
                                  callback = self.test,
                                  check    = True)
                dpg.add_menu_item(label = "Save", 
                                  callback = self.save_callback,
                                  check    = True)
                dpg.add_menu_item(label = "Load", 
                                  callback = lambda: dpg.show_item(self.tag_checkpoint_dialog),
                                  check    = True)
                dpg.add_menu_item(label = "Stop", 
                                  callback = self.stop_callback,
                                  check    = True)
            
            #PARAMETRI SECTION   
            with dpg.collapsing_header(label="Algorithm parameters", default_open=True):       
                with dpg.tab_bar():
                    with dpg.tab(label="FedAvg"):
                        with dpg.child_window(height=self.parameter_section_height):   
                            self.tag_num_clients      = dpg.add_input_int(label = "N  (num. of clients)",
                                                                          default_value = 100,
                                                                          width         = self.width/2)
                        
                            self.tag_training_rounds  = dpg.add_input_int(label = "R  (num. of training rounds)",
                                                                          default_value = 200,
                                                                          width         = self.width/2)
                        
                            self.tag_server_lr        = dpg.add_input_float(label = "LR (server learning rate)",
                                                                            default_value = 1.0,
                                                                            width         = self.width/2,
                                                                            min_value     = 0.001)
                        
                            self.tag_client_fraction  = dpg.add_slider_float(label = "C  (fraction of clients)",
                                                                             default_value = 0.1,
                                                                             width         = self.width/2,
                                                                             min_value     = 0.0,
                                                                             max_value     = 1.0)
                        
                            self.tag_client_epochs    = dpg.add_input_int(label = "E  (num. of client epochs)",
                                                                          default_value = 1,
                                                                          width         = self.width/2)
                        
                            with dpg.group():                              
                                self.tag_client_batch_size = dpg.add_input_float(label         = 'B  (client batch size (% of data))',
                                                                                 default_value = 0.02765,
                                                                                 width         = self.width/2,
                                                                                 callback      = self.update_number_of_steps)
                                
                                self.tag_client_number_of_steps = dpg.add_input_int(label         = 'K  (client number of steps)',
                                                                                    default_value = 25,
                                                                                    width         = self.width/2,
                                                                                    callback      = self.update_batch_size)
                                
                                self.update_number_of_steps()
                        
                            self.tag_client_lr        = dpg.add_input_float(label         = "LR (client learning rate)",
                                                                            default_value = 0.001,
                                                                            width         = self.width/2,
                                                                            min_value     = 0.001)
                        
                            self.tag_client_momentum  = dpg.add_input_float(label         = "Momentum (client learning momentum)",
                                                                            default_value = 0.0,
                                                                            width         = self.width/2,
                                                                            show          = True)

                    with dpg.tab(label="FedProx"):
                        with dpg.child_window(height=self.parameter_section_height):    
                            self.tag_client_mu = dpg.add_input_float(label = "Mu (client prox term)",
                                                                     default_value = 0.001,
                                                                     width         = self.width/2)

                    with dpg.tab(label="FedKernelAvg", show=False):    
                        with dpg.child_window(height=self.parameter_section_height):      
                            self.tag_kernel_radius_auto = dpg.add_checkbox(label="Auto adjust (min+max)")
                            
                            self.tag_kernel_type = dpg.add_combo(["Gaussian", "Poly6", "Spiky"],
                                                                 label         = "Kernel type",
                                                                 default_value = "Gaussian",
                                                                 width         = self.width/2,
                                                                 callback      = self.slider_drag_callback)
                            
                            self.tag_kernel_radius = dpg.add_slider_float(label = "h (kernel radius)",
                                                                          default_value = 1.0,
                                                                          min_value     = 0.1,
                                                                          max_value     = 2.0,
                                                                          callback      = self.slider_drag_callback,
                                                                          width         = self.width/2)
                            
                            self.tag_plot_kernel = dpg.add_plot(label = "Kernel",
                                                            height       = int(self.plot_height*0.5),
                                                            width        = int(self.width*0.5),
                                                            crosshairs   = True,
                                                            query        = True,
                                                            anti_aliased = True) 
                            
                            dpg.add_plot_legend(location = dpg.mvPlot_Location_East,
                                                outside  = False,
                                                parent   = self.tag_plot_kernel)
                            
                            self.tag_x_axis_kernel  = dpg.add_plot_axis(dpg.mvXAxis,
                                                                        label  = "r",
                                                                        parent = self.tag_plot_kernel)
                            self.tag_y_axis_kernel = dpg.add_plot_axis(dpg.mvYAxis, 
                                                                       label  = "r",
                                                                       parent = self.tag_plot_kernel)

                            self.tag_series_kernel  = dpg.add_line_series([], [],
                                                                          label  = "kernel",
                                                                          parent = self.tag_y_axis_kernel)
                            
                            self.slider_drag_callback()
                            

                            
                    with dpg.tab(label="FedVARP"):
                        with dpg.child_window(height=self.parameter_section_height):    
                            self.tag_clustering   = dpg.add_checkbox(label="Clustering")
                            
                            self.tag_num_clusters = dpg.add_input_int(label = "Number of clusters",
                                                                      default_value = 20,
                                                                      width         = self.width/2)
                            
                    with dpg.tab(label="FedRolex"):
                        with dpg.child_window(height=self.parameter_section_height):    
                            self.tag_min_model_size  = dpg.add_slider_double(label = "Min model size",
                                                                             default_value = 0.3,
                                                                             min_value     = 0,
                                                                             max_value     = 1.0,
                                                                             width         = self.width/2)
                            self.tag_max_model_size  = dpg.add_slider_double(label = "Max model size",
                                                                             default_value = 1.0,
                                                                             min_value     = 0,
                                                                             max_value     = 1.0,
                                                                             width         = self.width/2)
                            self.tag_same_model_size = dpg.add_checkbox(label="Use same size for all (max size)")
                    
                    with dpg.tab(label="Simulator"):
                        with dpg.child_window(height=self.parameter_section_height):    
                            self.tag_num_threads  = dpg.add_input_int(label = "Number of threads",
                                                                      default_value = 5,
                                                                      width         = self.width/2)
                            
                            self.tag_dataset      = dpg.add_combo(["MNIST", "FMNIST", "CIFAR10", "Shakespeare", "EMNIST Digits"],
                                                                  label         = "Dataset",
                                                                  default_value = "MNIST",
                                                                  width         = self.width/2)
                            
                            self.tag_model_type   = dpg.add_combo(["CNN", "GarmentClassifier", "LSTM", "CIFAR10Classifier", "LeNet5"],
                                                                  label         = "Model",
                                                                  default_value = "CNN",
                                                                  width         = self.width/2)
                            
                            self.tag_device       = dpg.add_combo(["cpu", "cuda"],
                                                                  label         = "Device", 
                                                                  default_value = "cpu",
                                                                  width         = self.width/2)
                            
                            self.tag_split_type   = dpg.add_combo(["iid", "random_ratios", "noniid", "dirichlet"],
                                                                  label         = "Split",
                                                                  default_value = "noniid",
                                                                  width         = self.width/2)
                            
                            self.tag_dirichlet_concentration = dpg.add_input_float(label = "Dirichlet concentration",
                                                                                 default_value = 1.0,
                                                                                 width         = self.width/2)
                            
                            self.tag_split_mincls = dpg.add_input_int(label = "Minimum number of classes per client",
                                                                      default_value = 2,
                                                                      width         = self.width/2)
                            
                            self.tag_plot_data_distribution = dpg.add_checkbox(label = "Plot data distribution histograms",
                                                                               default_value = False)
                            
                            self.tag_plot_avg_window_size   = dpg.add_input_int(label = "Avg plot window size",
                                                                                default_value = 5,
                                                                                width         = self.width/2)
                              
                            self.tag_checkpoint_interval    = dpg.add_input_int(label = "Checkpoint interval(rounds)",
                                                                                default_value = 5,
                                                                                width         = self.width/2)
            
            #PLOTS SECTION                
            with dpg.collapsing_header(label="Global model metrics", default_open=True):
                with dpg.group(horizontal=True):
                    self.tag_progress_bar = dpg.add_progress_bar(width=int(self.width*0.77))
                    dpg.add_text("Testing")
                    dpg.add_button(label    = "Sync plots",
                                   callback = self.sync_plots)
                       
                #ACCURACY PLOT
                self.tag_plot_acc = dpg.add_plot(label = "global model accuracy",
                                                 height       = self.plot_height,
                                                 width        = int(self.width*0.97),
                                                 crosshairs   = True,
                                                 query        = True,
                                                 anti_aliased = True) 
                   
                dpg.add_plot_legend(location = dpg.mvPlot_Location_East,
                                    outside  = False,
                                    parent   = self.tag_plot_acc)
                
                self.tag_x_axis_acc          = dpg.add_plot_axis(dpg.mvXAxis,
                                                                 label  = "round",
                                                                 parent = self.tag_plot_acc)
                self.tag_y_axis_acc          = dpg.add_plot_axis(dpg.mvYAxis, 
                                                                 label  = "accuracy",
                                                                 parent = self.tag_plot_acc)
                self.tag_series_acc          = dpg.add_line_series([], [],
                                                                   label  = "acc",
                                                                   parent = self.tag_y_axis_acc)
                self.tag_series_acc_avg      = dpg.add_line_series([], [],
                                                                   label  = "avg",
                                                                   parent = self.tag_y_axis_acc)
                self.tag_target_acc_dragline = dpg.add_drag_line(label = "target accuracy",
                                                                 color         = [255, 255, 255, 255],
                                                                 vertical      = False,
                                                                 default_value = 1.0,
                                                                 parent = self.tag_plot_acc)
                
                dpg.bind_item_theme(self.tag_series_acc, realtime_line_theme)
                dpg.bind_item_theme(self.tag_series_acc_avg, avg_line_theme)
                
                
                #LOSS PLOT      
                self.tag_plot_loss = dpg.add_plot(label = "global model loss",
                                                  height       = self.plot_height,
                                                  width        = int(self.width*0.97),
                                                  crosshairs   = True,
                                                  query        = True,
                                                  anti_aliased = True) 
                   
                dpg.add_plot_legend(location = dpg.mvPlot_Location_East,
                                    outside  = False,
                                    parent   = self.tag_plot_loss)
                
                self.tag_x_axis_loss          = dpg.add_plot_axis(dpg.mvXAxis,
                                                                  label  = "round",
                                                                  parent = self.tag_plot_loss)
                self.tag_y_axis_loss          = dpg.add_plot_axis(dpg.mvYAxis,
                                                                  label  = "loss",
                                                                  parent = self.tag_plot_loss)
                self.tag_series_loss          = dpg.add_line_series([], [],
                                                                    label  = "loss",
                                                                    parent = self.tag_y_axis_loss)
                self.tag_series_loss_avg      = dpg.add_line_series([], [],
                                                                    label  = "avg",
                                                                    parent = self.tag_y_axis_loss)
                self.tag_target_loss_dragline = dpg.add_drag_line(label = "target loss",
                                                                  color         = [255, 255, 255, 255],
                                                                  vertical      = False,
                                                                  default_value = 0,
                                                                  parent        = self.tag_plot_loss)
                
                dpg.bind_item_theme(self.tag_series_loss, realtime_line_theme)
                dpg.bind_item_theme(self.tag_series_loss_avg, avg_line_theme)
                    
            with dpg.window(label  = f"{self.id} Client List",
                            pos    = [self.width+15, 10],
                            height = int(self.height),
                            width  = 600):
                    self.tag_client_table = dpg.add_table(header_row     = True,
                                                          resizable      = True,
                                                          policy         = dpg.mvTable_SizingFixedFit,
                                                          borders_outerH = True,
                                                          borders_innerV = True,
                                                          borders_innerH = True,
                                                          borders_outerV = True)
                    self.tags_client_table_columns = []
        
        self.apply_ui_parameters()

    def apply_ui_parameters(self):
        #server
        self.num_clients       = dpg.get_value(self.tag_num_clients)
        self.training_rounds   = dpg.get_value(self.tag_training_rounds)
        self.server_lr         = dpg.get_value(self.tag_server_lr)
        self.client_fraction   = dpg.get_value(self.tag_client_fraction)
        
        #app
        if self.training_rounds > len(self.test_metrics):
            self.training_rounds = (self.training_rounds - len(self.test_metrics))         
        self.total_training_rounds = len(self.test_metrics) + self.training_rounds
        
        ##fedavg / scaffold / fedprox
        self.client_epochs     = dpg.get_value(self.tag_client_epochs)
        self.client_batch_size = dpg.get_value(self.tag_client_batch_size)
        self.client_lr         = dpg.get_value(self.tag_client_lr)
        self.client_momentum   = dpg.get_value(self.tag_client_momentum)
        self.update_number_of_steps()
        
        ##fedprox
        self.client_mu = dpg.get_value(self.tag_client_mu)
        
        ##fedvarp
        self.clustering   = dpg.get_value(self.tag_clustering)
        self.num_clusters = dpg.get_value(self.tag_num_clusters)
        
        ##fedscavar
        self.min_model_size   = dpg.get_value(self.tag_min_model_size)
        self.max_model_size   = dpg.get_value(self.tag_max_model_size)
        self.same_model_size  = dpg.get_value(self.tag_same_model_size)
        
        #fedkernelavg
        self.kernel_radius      = dpg.get_value(self.tag_kernel_radius)
        self.kernel_radius_auto = dpg.get_value(self.tag_kernel_radius_auto) 
        self.kernel_type        = dpg.get_value(self.tag_kernel_type)     
     
        ##simulation
        self.num_threads  = dpg.get_value(self.tag_num_threads)
        self.device       = dpg.get_value(self.tag_device)
        self.dataset      = dpg.get_value(self.tag_dataset) 
        self.model_type   = dpg.get_value(self.tag_model_type)
        self.split_type   = dpg.get_value(self.tag_split_type)
        self.dirichlet_concentration = dpg.get_value(self.tag_dirichlet_concentration)
        self.split_mincls = dpg.get_value(self.tag_split_mincls)
        self.plot_data_distribution = dpg.get_value(self.tag_plot_data_distribution)
        self.plot_avg_window_size   = dpg.get_value(self.tag_plot_avg_window_size)
        self.checkpoint_interval    = dpg.get_value(self.tag_checkpoint_interval)
        
        #self.slider_drop_callback()

        #if self.server != None:
        #    self.server.client_fraction = self.client_fraction
        #    self.server.training_rounds = self.training_rounds
        #    self.server.server_lr       = self.server_lr
        #    self.server.num_threads     = self.num_threads
        #    self.server.device          = self.device
            
        #    #TODO ClusterFedVARP cluster number je v konstruktorju

        #for c in self.clients:
        #    c.epochs = self.client_epochs
        #    c.batch_size = self.client_batch_size
        #    c.lr = self.client_lr
        #    c.momentum = self.client_momentum
        #    c.device = self.device
        #    dpg.set_value(c.tag_epochs, self.client_epochs)
        #    dpg.set_value(c.tag_batch_size, self.client_batch_size)
        #    dpg.set_value(c.tag_lr, self.client_lr)
        #    dpg.set_value(c.tag_momentum, self.client_momentum)
        #    dpg.set_value(c.tag_progress_bar, 0)        
        #    if c is FederatedClientProx:
        #        c.mu = self.client_mu
        #        dpg.set_value(c.tag_mu, self.client_mu)         

    def save_callback(self, sender, app_data):
        self.apply_ui_parameters()
        #self.post_round_callback(None)
        #self.plot_test_history()
        
    def load_callback(self, sender, app_data):
        #print(app_data["selections"])      
        filename, path = list(app_data["selections"].items())[0]
        self.load_model(path)
             
    def stop_callback(self, sender, app_data):
        self.server.stop_requested = True
        for c in self.clients:
            c.stop_requested = True
            
    def slider_drag_callback(self):
        kerneltype = dpg.get_value(self.tag_kernel_type)
        match kerneltype:
            case "Gaussian":
                dpg.set_value(self.tag_series_kernel, [[float(i/100) for i in range(0, 100)], [KernelAvg.gaussian_kernel(float(i/100), dpg.get_value(self.tag_kernel_radius)) for i in range(0, 100)]])          
       
            case "Poly6":
                dpg.set_value(self.tag_series_kernel, [[float(i/100) for i in range(0, 100)], [KernelAvg.poly6_kernel(float(i/100), dpg.get_value(self.tag_kernel_radius)) for i in range(0, 100)]])          
    
            case "Spiky":
                dpg.set_value(self.tag_series_kernel, [[float(i/100) for i in range(0, 100)], [KernelAvg.spiky_kernel(float(i/100), dpg.get_value(self.tag_kernel_radius)) for i in range(0, 100)]])          
    
        dpg.fit_axis_data(self.tag_x_axis_kernel)
        dpg.fit_axis_data(self.tag_y_axis_kernel)
        
    def update_number_of_steps(self):
        b = dpg.get_value(self.tag_client_batch_size)
        if b != 0:
            dpg.set_value(self.tag_client_number_of_steps, math.ceil(1 / b))
            
    def update_batch_size(self):
        k = dpg.get_value(self.tag_client_number_of_steps)
        if k != 0:
            dpg.set_value(self.tag_client_batch_size, 1 / k)
             
    def load_model(self, path:str=None):     
        if path != None:
            model_data = torch.load(path)
                
            self.test_metrics = model_data["test_metrics"]
            
            print(model_data["last_training_algo"], self.test_metrics)
            
            dpg.set_value(self.tag_num_clients, model_data["num_clients"])
            dpg.set_value(self.tag_training_rounds, model_data["training_rounds"])
            dpg.set_value(self.tag_server_lr, model_data["server_lr"])
            dpg.set_value(self.tag_client_fraction, model_data["client_fraction"])
            dpg.set_value(self.tag_client_epochs, model_data["client_epochs"])
            dpg.set_value(self.tag_client_batch_size, model_data["client_batch_size"])
            dpg.set_value(self.tag_client_lr, model_data["client_lr"])
            dpg.set_value(self.tag_client_momentum, model_data["client_momentum"])
            dpg.set_value(self.tag_client_mu, model_data["client_mu"])
            dpg.set_value(self.tag_clustering, model_data["clustering"])
            dpg.set_value(self.tag_num_clusters, model_data["num_clusters"])
            dpg.set_value(self.tag_num_threads, model_data["num_threads"])
            dpg.set_value(self.tag_device, model_data["device"])
            dpg.set_value(self.tag_dataset, model_data["dataset"])
            dpg.set_value(self.tag_model_type, model_data["model_type"])
            dpg.set_value(self.tag_split_type, model_data["split_type"])
            dpg.set_value(self.tag_split_mincls, model_data["split_mincls"])
            dpg.set_value(self.tag_plot_data_distribution, model_data["plot_data_distribution"])
            dpg.set_value(self.tag_min_model_size, model_data["min_model_size"])
            dpg.set_value(self.tag_max_model_size, model_data["max_model_size"])
            dpg.set_value(self.tag_same_model_size, model_data["same_model_size"])
            
            if "dirichlet_concentration" in model_data:
                dpg.set_value(self.tag_dirichlet_concentration, model_data["dirichlet_concentration"])
                
            self.apply_ui_parameters()
            
            if "FederatedServerAvg+FederatedClientAvg" in model_data["last_training_algo"]:
                self.deploy_fedavg()
                
            if "FederatedServerAvg+FederatedClientProx" in model_data["last_training_algo"]:
                self.deploy_fedprox()  
                
            if "FederatedServerSCAFFOLD" in model_data["last_training_algo"]:
                self.deploy_scaffold()
                self.server.control = model_data["scaffold_control"]
                
            if "FederatedServerVARP" in model_data["last_training_algo"]:
                self.deploy_fedvarp()
                self.server.clients_y = model_data["fedvarp_clients_y"]
                
            if "FederatedServerClusterVARP" in model_data["last_training_algo"]:
                self.deploy_fedvarp()
                self.server.cluster_y       = model_data["clusterfedvarp_cluster_y"]
                self.server.cluster_indices = model_data["clusterfedvarp_cluster_indices"]
                
            if "FederatedServerSCAVAR" in model_data["last_training_algo"]:
                self.deploy_scavar()
                self.server.control         = model_data["fedscavar_control"]
                self.server.cluster_y       = model_data["fedscavar_cluster_y"]
                self.server.cluster_indices = model_data["fedscavar_cluster_indices"]
            
            if "KernelAvg" in model_data["last_training_algo"]:
                dpg.set_value(self.tag_kernel_radius, model_data["kernel_radius"])
                dpg.set_value(self.tag_kernel_radius_auto, model_data["kernel_radius_auto"])
                if 'kernel_type' in model_data:
                    dpg.set_value(self.tag_kernel_type, model_data["kernel_type"])
                self.deploy_kernelavg()
                       
            if "FederatedServerRolex" in model_data["last_training_algo"]:
                self.deploy_fedrolex()
               
            self.model.load_state_dict(model_data["global_model_state_dict"])
            
            dpg.set_axis_limits(self.tag_x_axis_acc, -3.0, len(self.test_metrics)  + 3)
            dpg.set_axis_limits(self.tag_x_axis_loss, -3.0, len(self.test_metrics) + 3)
            dpg.set_axis_limits(self.tag_y_axis_acc, -0.1, 1.1)
            
            for i in range(10000000):continue
            dpg.set_axis_limits_auto(self.tag_x_axis_acc)
            
            self.plot_test_history()
        else:   
            match self.model_type:
                case "GarmentClassifier":
                    self.model = GarmentClassifier()
                case "CNN":
                    self.model = CNN()
                case "CNNNet":
                    self.model = CNN()
                case "CIFAR10Classifier":
                    self.model = CIFAR10Net()
                case "LSTM":
                    self.model = CharLSTM()
                case "LeNet5":
                    self.model = LeNet5()
            
    def load_dataset(self):
        match self.dataset:        
            case "FMNIST":
                transform                = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
                self.local_training_data = torchvision.datasets.FashionMNIST('./data', train=True , transform=transform, download=True)
                self.local_test_data     = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)
            case "MNIST":
                transform                = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
                self.local_training_data = torchvision.datasets.MNIST('./data', train=True , transform=transform, download=True)
                self.local_test_data     = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=True)
            case "CIFAR10":
                transform                = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                self.local_training_data = torchvision.datasets.CIFAR10('./data', train=True , transform=transform, download=True)
                self.local_test_data     = torchvision.datasets.CIFAR10('./data', train=False, transform=transform, download=True)
            case "Shakespeare":
                self.local_training_data = Shakespeare(train=True)
                self.local_test_data     = Shakespeare(train=False)
            case "EMNIST Digits":
                transform                = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
                self.local_training_data = torchvision.datasets.EMNIST('./data', "digits", train=True , transform=transform, download=True)
                self.local_test_data     = torchvision.datasets.EMNIST('./data', "digits", train=False, transform=transform, download=True)
        
        if self.client_batch_size > len(self.local_test_data):
            self.client_batch_size = len(self.local_test_data)
            
        self.testing_loader = torch.utils.data.DataLoader(self.local_test_data, batch_size=32, shuffle=True)
        print(self.testing_loader)
        
        if self.dataset == "Shakespeare": #Shakespeare je ze noniid ga samo enakomerno razdelimo
            self.num_clients = len(self.local_training_data.get_client_dic().keys())
            dpg.set_value(self.tag_num_clients, self.num_clients)
            self.datasets = [Subset(self.local_training_data, list(indices)) for indices in self.local_training_data.get_client_dic().values()]
            
            #import matplotlib.pyplot as plt
            #plt.bar(np.arange(0, len(self.datasets), 1), [len(d) for d in self.datasets])
            #plt.show()
            #self.datasets = dataset_split2(self.local_training_data,
            #                               self.num_clients,
            #                               "iid",
            #                               self.split_mincls,
            #                               plot=self.plot_data_distribution) 
        else:
            self.datasets = dataset_split2(self.local_training_data,
                                           self.num_clients,
                                           self.split_type,
                                           self.split_mincls,
                                           plot=self.plot_data_distribution,
                                           cncntrcn=self.dirichlet_concentration)

    def prepare_client_list_columns(self, column_names, column_widths):
        for tag in self.tags_client_table_columns:
            dpg.delete_item(tag)       
        self.tags_client_table_columns.clear() 
        
        for column_name, column_width in zip(column_names, column_widths):
            self.tags_client_table_columns.append(dpg.add_table_column(label                = column_name,
                                                                       init_width_or_weight = column_width,
                                                                       parent               = self.tag_client_table))
        
    def deploy_fedavg(self):
        self.apply_ui_parameters()
        self.clear_client_list()     
        self.load_dataset()
        self.prepare_client_list_columns(["Client", "Progress", "Accuracy", "Loss", "Total loss"], [100, 160, 90, 90, 80])

        if self.model == None:
            self.load_model()

        for k in range(self.num_clients):
            client = FederatedClientAvg(f'client_{k}', self.datasets[k],
                                        self.client_epochs, self.client_batch_size, self.client_lr,
                                        self.client_momentum, self.device, parent_tag=self.tag_client_table)
            self.clients.append(client)
        
        self.server = FederatedServerAvg(self.model, self.clients, self.training_rounds, self.server_lr,
                                         self.client_fraction, self.num_threads, self.post_round_callback, self.device)
        
        #import matplotlib.pyplot as plt
        #
        #plt.bar(np.arange(0, len(self.clients),1), [len(c) for c in self.clients])
        #plt.show()
        
        dpg.configure_item(self.id, label=f'{self.id} {self.server.__class__.__name__}')
            
    def deploy_fedprox(self):
        self.apply_ui_parameters()
        self.clear_client_list()
        self.load_dataset() 
        self.prepare_client_list_columns(["Client", "Progress", "Accuracy", "Loss", "Total loss"], [100, 160, 90, 90, 80])
        
        if self.model == None:
            self.load_model()
         
        for k in range(self.num_clients):
            client = FederatedClientProx(f'client_{k}', self.datasets[k],
                                         self.client_epochs, self.client_batch_size, self.client_lr,
                                         self.client_momentum, self.client_mu, self.device, parent_tag=self.tag_client_table)
            self.clients.append(client)
                
        self.server = FederatedServerAvg(self.model, self.clients, self.training_rounds, self.server_lr,
                                         self.client_fraction, self.num_threads, self.post_round_callback, self.device) 
        
        dpg.configure_item(self.id, label=f'{self.id} {self.server.__class__.__name__}')
        
    def deploy_scaffold(self):
        self.apply_ui_parameters()
        self.clear_client_list()
        self.load_dataset()
        self.prepare_client_list_columns(["Client", "Progress", "Accuracy", "Loss", "Total loss"], [100, 160, 90, 90, 80])
        
        if self.model == None:
            self.load_model()
                
        for k in range(self.num_clients):
            client = FederatedClientSCAFFOLD(f'client_{k}', self.datasets[k],
                                             self.client_epochs, self.client_batch_size, self.client_lr,
                                             self.client_momentum, self.device, parent_tag=self.tag_client_table)
            self.clients.append(client)
        
        self.server = FederatedServerSCAFFOLD(self.model, self.clients, self.training_rounds, self.server_lr,
                                              self.client_fraction, self.num_threads, self.post_round_callback, self.device)
        
        dpg.configure_item(self.id, label=f'{self.id} {self.server.__class__.__name__}')   
        
    def deploy_fedvarp(self):
        self.apply_ui_parameters()
        self.clear_client_list()     
        self.load_dataset()
        self.prepare_client_list_columns(["Client", "Progress", "Accuracy", "Loss", "Total loss"], [100, 160, 90, 90, 80])
        
        if self.model == None:
            self.load_model()
               
        for k in range(self.num_clients):
            client = FederatedClientAvg(f'client_{k}', self.datasets[k],
                                        self.client_epochs, self.client_batch_size, self.client_lr,
                                        self.client_momentum, self.device, parent_tag=self.tag_client_table)
            self.clients.append(client)
        
        if self.clustering:
            self.server = FederatedServerClusterVARP(self.model, self.clients, self.training_rounds, self.server_lr,
                                                     self.client_fraction, self.num_clusters, self.num_threads, self.post_round_callback, self.device)
            
            dpg.configure_item(self.id, label=f'{self.id} FederatedServerClusterVARP') 
        else:
            self.server = FederatedServerVARP(self.model, self.clients, self.training_rounds, self.server_lr,
                                              self.client_fraction, self.num_threads, self.post_round_callback, self.device)   
                 
            dpg.configure_item(self.id, label=f'{self.id} {self.server.__class__.__name__}')  
             
    def deploy_fedrolex(self):
        self.apply_ui_parameters()
        self.clear_client_list()
        self.load_dataset()
        self.prepare_client_list_columns(["Client", "Progress", "Accuracy", "Loss", "Total loss", "Size"], [100, 100, 90, 90, 80, 70])

        if self.model == None:
            self.load_model()
                
        model_fraction = self.max_model_size
        for k in range(self.num_clients):
            client = FederatedClientRolex(f'client_{k}', self.datasets[k], model_fraction,
                                          self.client_epochs, self.client_batch_size, self.client_lr,
                                          self.client_momentum, self.device, parent_tag=self.tag_client_table)
            self.clients.append(client)
            if not self.same_model_size:
                model_fraction = random.uniform(self.min_model_size, self.max_model_size)
    
        self.server = FederatedServerRolex(self.model, self.clients, self.training_rounds, self.server_lr,
                                           self.client_fraction, self.num_threads, self.post_round_callback, self.device)
        
        #import matplotlib.pyplot as plt
        #
        #plt.bar(np.arange(0, len(self.clients),1), sorted([c.model_fraction for c in self.clients]))
        #plt.show()
        
        dpg.configure_item(self.id, label=f'{self.id} {self.server.__class__.__name__}')
               
    def deploy_scavar(self):
        self.apply_ui_parameters()
        self.clear_client_list()
        self.load_dataset()
        self.prepare_client_list_columns(["Client", "Progress", "Accuracy", "Loss", "Total loss", "Size"], [100, 100, 90, 90, 80, 70])

        if self.model == None:
            self.load_model()
  
        model_fraction = self.max_model_size
        for k in range(self.num_clients):
            client = FederatedClientSCAVAR(f'client_{k}', self.datasets[k], model_fraction,
                                           self.client_epochs, self.client_batch_size, self.client_lr,
                                           self.client_momentum, self.client_mu, self.device, parent_tag=self.tag_client_table)
            self.clients.append(client)
            if not self.same_model_size:
                model_fraction = random.uniform(self.min_model_size, self.max_model_size)
        
        self.server = FederatedServerSCAVAR(self.model, self.clients, self.training_rounds, self.server_lr,
                                            self.client_fraction, self.num_clusters, self.num_threads, self.post_round_callback, self.device)
        dpg.configure_item(self.id, label=f'{self.id} {self.server.__class__.__name__}')

    def deploy_kernelavg(self):
        self.apply_ui_parameters()
        self.clear_client_list()
        self.load_dataset()
        self.prepare_client_list_columns(["Client", "Progress", "Accuracy", "Loss", "Total loss"], [100, 160, 90, 90, 80])

        if self.model == None:
            self.load_model()
  
        for k in range(self.num_clients):
            client = FederatedClientAvg(f'client_{k}', self.datasets[k],
                                        self.client_epochs, self.client_batch_size, self.client_lr,
                                        self.client_momentum, self.device, parent_tag=self.tag_client_table)
            self.clients.append(client)
        
        self.server = KernelAvg(self.model, self.clients, self.training_rounds, self.server_lr,
                                self.client_fraction, self.num_threads, self.kernel_radius, self.kernel_radius_auto, self.post_round_callback, self.device)
        dpg.configure_item(self.id, label=f'{self.id} {self.server.__class__.__name__}')    
                             
    def fit(self):
        self.apply_ui_parameters()
        
        self.tag_dragline_acc_start = dpg.add_drag_line(label         = self.server.__class__.__name__,
                                                        color         = [150, 255, 0, 70],
                                                        default_value = len(self.test_metrics),
                                                        parent        = self.tag_plot_acc)
        
        self.tag_dragline_loss_start = dpg.add_drag_line(label         = self.server.__class__.__name__,
                                                         color         = [150, 255, 0, 70],
                                                         default_value = len(self.test_metrics),
                                                         parent        = self.tag_plot_loss)
        
        self.tag_dragline_acc_end = dpg.add_drag_line(label         = "end", 
                                                      color         = [255, 150, 0, 70],
                                                      default_value = len(self.test_metrics) + self.training_rounds,
                                                      parent        = self.tag_plot_acc)
        
        self.tag_dragline_loss_end = dpg.add_drag_line(label         = "end", 
                                                       color         = [255, 150, 0, 70],
                                                       default_value = len(self.test_metrics) + self.training_rounds,
                                                       parent        = self.tag_plot_loss)
        
        dpg.set_axis_limits(self.tag_x_axis_acc, -3.0, len(self.test_metrics) + self.training_rounds + 3)
        dpg.set_axis_limits(self.tag_x_axis_loss, -3.0, len(self.test_metrics) + self.training_rounds + 3)
        dpg.set_axis_limits(self.tag_y_axis_acc, -0.1, 1.1)
        dpg.set_axis_limits(self.tag_y_axis_loss, -0.3, 3.3)
        
        for i in range(10000000):continue
        dpg.set_axis_limits_auto(self.tag_x_axis_acc)
        dpg.set_axis_limits_auto(self.tag_x_axis_loss)
        
        for i in range(len(self.test_metrics)):
            random.sample(range(self.num_clients), max(int(self.num_clients * self.client_fraction), 1))  
        
        def tfit():
            if len(self.test_metrics) == 0:
                self.test()
            self.server.fit()
            
        self.server.stop_requested = False    
        self.thread = threading.Thread(target = tfit)
        self.thread.start()
     
    def test(self):
        self.model.to(self.device)
        self.model.eval()

        loss_fn = nn.CrossEntropyLoss()
        
        self.test_metrics.reset()   
           
        with torch.no_grad():
            for batch_idx, (batch_inputs, batch_labels) in enumerate(self.testing_loader):
                batch_inputs, batch_labels = batch_inputs.to(self.device), batch_labels.to(self.device)

                outputs = self.model(batch_inputs)

                loss = loss_fn(outputs, batch_labels)
                
                self.test_metrics.update(outputs, batch_labels, loss)
           
                if batch_idx % 20 == 0 or batch_idx == len(self.testing_loader) - 1:
                    dpg.set_value(self.tag_progress_bar, batch_idx / max(len(self.testing_loader) - 1, 1))
      
            self.test_metrics.archive_current()
             
        dpg.set_value(self.tag_progress_bar, 0)
        
        self.apply_ui_parameters()  
        self.plot_test_history()

    def plot_test_history(self):
        x = [i for i in range(len(self.test_metrics))]
                
        N = self.plot_avg_window_size
        y_padded = np.pad(self.test_metrics.history_accuracy, (N//2, N-1-N//2), mode='edge')
        acc_avg  = np.convolve(y_padded, np.ones((N,))/N, mode='valid') 
        y_padded = np.pad(self.test_metrics.history_loss, (N//2, N-1-N//2), mode='edge')
        loss_avg = np.convolve(y_padded, np.ones((N,))/N, mode='valid') 
        
        dpg.set_value(self.tag_series_acc_avg, [x, acc_avg])
        dpg.set_item_label(self.tag_series_acc_avg, f"avg: {acc_avg[-1]:.04f}")
        dpg.set_value(self.tag_series_loss_avg, [x, loss_avg])
        dpg.set_item_label(self.tag_series_loss_avg, f"avg : {loss_avg[-1]:.04f}")
   
        dpg.set_value(self.tag_series_acc, [x, self.test_metrics.history_accuracy])
        dpg.set_item_label(self.tag_series_acc, f"acc: {self.test_metrics.history_accuracy[-1]:.04f}")
        
        dpg.set_value(self.tag_series_loss, [x, self.test_metrics.history_loss])
        dpg.set_item_label(self.tag_series_loss, f"loss: {self.test_metrics.history_loss[-1]:.04f}")
        dpg.set_axis_limits(self.tag_y_axis_loss, -0.2, max(self.test_metrics.history_loss) + 0.2)       
  
        self.clear_dragpoints()
               
        for i, acc in enumerate(self.test_metrics.history_accuracy):
            if max(self.test_metrics.history_accuracy) == acc:
                self.tag_dragpoints.append(dpg.add_drag_point(label         = "checkpoint",
                                                              default_value = (i, acc),
                                                              color         = [255, 0, 100],
                                                              parent        = self.tag_plot_acc))
        for i, loss in enumerate(self.test_metrics.history_loss):
            if min(self.test_metrics.history_loss) == loss:
                self.tag_dragpoints.append(dpg.add_drag_point(label         = "checkpoint",
                                                              default_value = (i, loss),
                                                              color         = [255, 0, 100],
                                                              parent        = self.tag_plot_loss))
    
    def sync_plots(self):
        dpg.set_axis_limits(self.tag_x_axis_loss,
                            dpg.get_axis_limits(self.tag_x_axis_acc)[0],
                            dpg.get_axis_limits(self.tag_x_axis_acc)[1])
        for i in range(10000000):continue
        dpg.set_axis_limits_auto(self.tag_x_axis_loss)
          
    def clear_client_list(self):
        for c in self.clients:
            dpg.delete_item(c.tag_table_row)
        self.clients.clear() 
    
    def clear_dragpoints(self):
        for t in self.tag_dragpoints:
            dpg.delete_item(t)   
        self.tag_dragpoints.clear()
              
    def post_round_callback(self, server):
        if self.model != None:
            self.test()
            
            if (max(self.test_metrics.history_accuracy) == self.test_metrics.history_accuracy[-1] or 
                min(self.test_metrics.history_loss)     == self.test_metrics.history_loss[-1]     or 
                (len(self.test_metrics) > 0 and len(self.test_metrics) % self.checkpoint_interval == 0)):
                save_data = {
                    'global_model_state_dict': self.model.state_dict(),
                    'test_metrics'           : self.test_metrics,      
                    'num_clients'            : dpg.get_value(self.tag_num_clients),
                    'training_rounds'        : self.total_training_rounds,
                    'server_lr'              : dpg.get_value(self.tag_server_lr),
                    'client_fraction'        : dpg.get_value(self.tag_client_fraction),
                    'client_epochs'          : dpg.get_value(self.tag_client_epochs),
                    'client_batch_size'      : dpg.get_value(self.tag_client_batch_size),
                    'client_lr'              : dpg.get_value(self.tag_client_lr),
                    'client_momentum'        : dpg.get_value(self.tag_client_momentum),
                    'client_mu'              : dpg.get_value(self.tag_client_mu),
                    'clustering'             : dpg.get_value(self.tag_clustering),
                    'num_clusters'           : dpg.get_value(self.tag_num_clusters),
                    'num_threads'            : dpg.get_value(self.tag_num_threads),
                    'device'                 : dpg.get_value(self.tag_device),
                    'dataset'                : dpg.get_value(self.tag_dataset) ,
                    'model_type'             : dpg.get_value(self.tag_model_type),
                    'split_type'             : dpg.get_value(self.tag_split_type),
                    'dirichlet_concentration': dpg.get_value(self.tag_dirichlet_concentration),
                    'split_mincls'           : dpg.get_value(self.tag_split_mincls),
                    'plot_data_distribution' : dpg.get_value(self.tag_plot_data_distribution),
                    'min_model_size'         : dpg.get_value(self.tag_min_model_size),
                    'max_model_size'         : dpg.get_value(self.tag_max_model_size),
                    'same_model_size'        : dpg.get_value(self.tag_same_model_size),
                    'kernel_radius'          : dpg.get_value(self.tag_kernel_radius),
                    'kernel_type'            : dpg.get_value(self.tag_kernel_type),
                    'kernel_radius_auto'     : dpg.get_value(self.tag_kernel_radius_auto),
                    'last_training_algo'     : f"{self.server.__class__.__name__}+{self.clients[0].__class__.__name__}"
                }

                if "FederatedServerSCAFFOLD" in save_data["last_training_algo"]:
                    save_data.update({"scaffold_control": self.server.control})
                    
                if "FederatedServerVARP" in save_data["last_training_algo"]:
                    save_data.update({"fedvarp_clients_y": self.server.clients_y})
                   
                if "FederatedServerClusterVARP" in save_data["last_training_algo"]:   
                    save_data.update({"clusterfedvarp_cluster_y": self.server.cluster_y})
                    save_data.update({"clusterfedvarp_cluster_indices": self.server.cluster_indices})

                if "FederatedServerSCAVAR" in save_data["last_training_algo"]:   
                    save_data.update({"fedscavar_control": self.server.control})
                    save_data.update({"fedscavar_cluster_y": self.server.cluster_y})
                    save_data.update({"fedscavar_cluster_indices": self.server.cluster_indices})
                            
                if not os.path.exists(self.checkpoint_folder):
                    os.makedirs(self.checkpoint_folder)   
                    
                torch.save(save_data, f'{self.checkpoint_folder}/{self.server.__class__.__name__}_{self.dataset}_{dpg.get_value(self.tag_model_type)}_r{len(self.test_metrics)}_acc_{self.test_metrics.history_accuracy[-1]:.4f}_loss_{self.test_metrics.history_loss[-1]:.4f}.pt')
                
            if (dpg.get_value(self.tag_target_acc_dragline) <= self.test_metrics.history_accuracy[-1] or 
                dpg.get_value(self.tag_target_loss_dragline) >= self.test_metrics.history_loss[-1]):
                self.server.stop_requested = True
            if (math.ceil(dpg.get_value(self.tag_dragline_acc_end)) <= len(self.test_metrics) or 
                dpg.get_value(self.tag_dragline_loss_end) <= len(self.test_metrics)):
                self.server.stop_requested = True


