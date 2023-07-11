from collections import OrderedDict
import numpy as np
from src.utils import EpochMetrics
from matplotlib import pyplot as plt
import torch

def single_comparsion():
    checkpoints_diri  = {
        "FedSGD 0.01"          : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/fedsgd_c_0.01.pt',
        "FedSGD"               : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/fedsgd.pt',
        "FedAvg"               : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedavg.pt',
        "FedProx Mu=0.25"      : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedprox_0.25.pt',
        "SCAFFOLD"             : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/scaffold.pt',
        "FedVARP"              : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedvarp.pt',
        "FedScavar C=100"      : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedscavar_100_clusters.pt',
        "ClusterFedVARP C=1"   : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/clustervarp_1_clusters.pt',
        "ClusterFedVARP C=10"  : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/clustervarp_10_clusters.pt',
        "FedScavar C=1"        : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedscavar_1_clusters.pt',
        "FedScavar C=10"       : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedscavar_10_clusters.pt',
        "FedRolex Random sizes": f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedrolex.pt',
        "FedScavar C=10 Rolex" : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedscavar_10_clusters_rolex.pt'
    }

    checkpoints_noniid  = {
        "FedSGD 0.01"          : f'./checkpoints/MNIST/LeNet5/noniid/fedsgd_c_0.01.pt',
        "FedSGD"               : f'./checkpoints/MNIST/LeNet5/noniid/fedsgd.pt',
        "FedAvg"               : f'./checkpoints/MNIST/LeNet5/noniid/1x10/fedavg.pt',
        "FedProx Mu=0.25"      : f'./checkpoints/MNIST/LeNet5/noniid/1x10/fedprox_0.25.pt',
        "SCAFFOLD"             : f'./checkpoints/MNIST/LeNet5/noniid/1x10/scaffold.pt',
        "FedVARP"              : f'./checkpoints/MNIST/LeNet5/noniid/1x10/fedvarp.pt',
        "FedScavar C=100"      : f'./checkpoints/MNIST/LeNet5/noniid/1x10/fedscavar_100_clusters.pt',
        "ClusterFedVARP C=1"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/clustervarp_1_clusters.pt',
        "FedScavar C=1"        : f'./checkpoints/MNIST/LeNet5/noniid/1x10/fedscavar_1_clusters.pt',
        "ClusterFedVARP C=10"  : f'./checkpoints/MNIST/LeNet5/noniid/1x10/clustervarp_10_clusters.pt',
        "FedScavar C=10"       : f'./checkpoints/MNIST/LeNet5/noniid/1x10/fedscavar_10_clusters.pt',
        "FedRolex Random sizes": f'./checkpoints/MNIST/LeNet5/noniid/1x10/fedrolex.pt',
        "FedScavar C=10 Rolex" : f'./checkpoints/MNIST/LeNet5/noniid/1x10/fedscavar_10_clusters_rolex.pt'
    }

    checkpoints_clusters  = {
        "ClusterFedVARP C=100" : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedvarp.pt',
        #"FedScavar C=100"      : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/1x10/fedscavar_100_clusters.pt',
        "ClusterFedVARP C=1"   : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/clustervarp_1_clusters.pt',
        "ClusterFedVARP C=10"  : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/clustervarp_10_clusters.pt',
        #"FedScavar C=1"        : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/1x10/fedscavar_1_clusters.pt',
        #"FedScavar C=10"       : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/1x10/fedscavar_10_clusters.pt',
    }

    checkpoints_rolex_epochs  = {
        "FedRolex E=1"             : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/1x10/fedrolex.pt',
        "FedScavar Rolex C=10 E=1" : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/1x10/fedscavar_10_clusters_rolex.pt',
        "FedRolex E=5"             : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/fedrolex.pt',
        "FedScavar Rolex C=10 E=5" : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/fedscavar_10_clusters_rolex.pt',
        "FedRolex E=20"             : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/20x10/fedrolex.pt',
        "FedScavar Rolex C=10 E=20" : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/20x10/fedscavar_10_clusters_rolex.pt'
    }

    checkpoints_shakespeare  = { 
        "FedAvg"               : f'./checkpoints/Shakespeare/dk_shakespeare/fedavg.pt',
        "FedProx Mu=0.001"     : f'./checkpoints/Shakespeare/dk_shakespeare/fedprox_0.001.pt',
        #"FedProx Mu=0.01"      : f'./checkpoints/Shakespeare/dk_shakespeare/fedprox_0.01.pt',
        "SCAFFOLD"             : f'./checkpoints/Shakespeare/dk_shakespeare/scaffold.pt',
        "FedVARP"              : f'./checkpoints/Shakespeare/dk_shakespeare/fedvarp.pt',
        "FedSCAVAR"            : f'./checkpoints/Shakespeare/dk_shakespeare/fedscavar_138_clusters.pt',
        "ClusterFedVARP C=1"   : f'./checkpoints/Shakespeare/dk_shakespeare/clustervarp_1_clusters.pt',
        "FedScavar C=1"        : f'./checkpoints/Shakespeare/dk_shakespeare/fedscavar_1_clusters.pt',
        "ClusterFedVARP C=10"  : f'./checkpoints/Shakespeare/dk_shakespeare/clustervarp_10_clusters.pt',
        "FedScavar C=10"       : f'./checkpoints/Shakespeare/dk_shakespeare/fedscavar_10_clusters.pt',
        "FedRolex Random sizes": f'./checkpoints/Shakespeare/dk_shakespeare/fedrolex.pt',
        "FedScavar C=10 Rolex" : f'./checkpoints/Shakespeare/dk_shakespeare/fedscavar_10_clusters_rolex.pt'
    }

    checkpoints_fractions  = {
        #"FedAvg C=0.01"  : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.01/fedavg.pt',
        #"FedAvg C=0.1"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/fedavg.pt',
        #"FedAvg C=0.5"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.5/fedavg.pt',
        "FedProx Mu=0.25 C=0.01"  : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.01/fedprox_0.25.pt',
        "FedProx Mu=0.25 C=0.1"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/fedprox_0.25.pt',
        "FedProx Mu=0.25 C=0.5"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.5/fedprox_0.25.pt',
        #"SCAFFOLD C=0.01"  : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.01/scaffold.pt',
        #"SCAFFOLD C=0.1"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/scaffold.pt',
        #"SCAFFOLD C=0.5"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.5/scaffold.pt',
    }

    model_datas = {name: torch.load(path) for name, path in checkpoints_diri.items()}

    fig, ((ax1), (ax2)) = plt.subplots(2, 1)

    def plot_model(model_data, label:str=None):
        if label == None:
            label = model_data["last_training_algo"] + f', N={model_data["num_clients"]}'
        else:
            label = label + f', N={model_data["num_clients"]}'
            
        ax1.plot(model_data["test_metrics"].history_accuracy, label=label + f', max acc={max(model_data["test_metrics"].history_accuracy):.4f}')
        ax2.plot(model_data["test_metrics"].history_loss    , label=label + f', min loss={min(model_data["test_metrics"].history_loss):.4f}')

    def steps_to_accuracy(model_data, target_accuracy):
        for i in range(len(model_data["test_metrics"])):
            if model_data["test_metrics"].history_accuracy[i] >= target_accuracy:
                return i, model_data["test_metrics"].history_accuracy[i] * 100
        return -1, -1

    for name, d in model_datas.items():
        plot_model(d, name)
        step, acc = steps_to_accuracy(d, 0.5)
        sgd_step, sgd_acc = steps_to_accuracy(model_datas["FedSGD"], 0.5)
        print("{:<30} {:<55} {:<10} {:<10} {:<15} {:<15} {:<15}".format(name, d["last_training_algo"], f'N={d["num_clients"]}', f'C={d["num_clusters"]}', f'step={step}', f'acc={acc:.2f}', f'norm={sgd_step / step}'))

    ax1.legend()
    ax2.legend()
    #plt.tight_layout()
    plt.show()

def plot_model2(ax, model_data, label:str=None, acc=True):
    if label == None:
        label = model_data["last_training_algo"] + f', N={model_data["num_clients"]}'
    else:
        label = label + f', N={model_data["num_clients"]}'
    
    if acc:
        ax.plot(model_data["test_metrics"].history_accuracy, label=label)
    else:
        ax.plot(model_data["test_metrics"].history_loss, label=label)

def four_plots_comparison():
    checkpoints_1  = {
        "SCAFFOLD"             : f'./checkpoints/MNIST/LeNet5/noniid/20x10/scaffold.pt',
        "FedVARP"              : f'./checkpoints/MNIST/LeNet5/noniid/20x10/fedvarp.pt',
        "FedScavar C=100"      : f'./checkpoints/MNIST/LeNet5/noniid/20x10/fedscavar_100_clusters.pt'
    }
    #checkpoints_1 = {
    #    "ClusterFedVARP C=100" : f'./checkpoints/MNIST/LeNet5/noniid/5x10/fedvarp.pt',
    #    "ClusterFedVARP C=1"   : f'./checkpoints/MNIST/LeNet5/noniid/5x10/clustervarp_1_clusters.pt',
    #    "ClusterFedVARP C=10"  : f'./checkpoints/MNIST/LeNet5/noniid/5x10/clustervarp_10_clusters.pt',
    #    "FedScavar C=1"        : f'./checkpoints/MNIST/LeNet5/noniid/5x10/fedscavar_1_clusters.pt',
    #    "FedScavar C=10"       : f'./checkpoints/MNIST/LeNet5/noniid/5x10/fedscavar_10_clusters.pt',
    #    "FedScavar C=100"      : f'./checkpoints/MNIST/LeNet5/noniid/5x10/fedscavar_100_clusters.pt',
    #}
    #checkpoints_1  = {
    #    "FedAvg"               : f'./checkpoints/MNIST/LeNet5/noniid/5x10/fedavg.pt',
    #    "FedProx Mu=0.25"      : f'./checkpoints/MNIST/LeNet5/noniid/5x10/fedprox_0.25.pt',
    #    "SCAFFOLD"             : f'./checkpoints/MNIST/LeNet5/noniid/5x10/scaffold.pt',
    #    "FedVARP"              : f'./checkpoints/MNIST/LeNet5/noniid/5x10/fedvarp.pt',
    #    "FedScavar C=100"      : f'./checkpoints/MNIST/LeNet5/noniid/5x10/fedscavar_100_clusters.pt',
    #    "ClusterFedVARP C=1"   : f'./checkpoints/MNIST/LeNet5/noniid/5x10/clustervarp_1_clusters.pt',
    #    "ClusterFedVARP C=10"  : f'./checkpoints/MNIST/LeNet5/noniid/5x10/clustervarp_10_clusters.pt',
    #    "FedScavar C=1"        : f'./checkpoints/MNIST/LeNet5/noniid/5x10/fedscavar_1_clusters.pt',
    #    "FedScavar C=10"       : f'./checkpoints/MNIST/LeNet5/noniid/5x10/fedscavar_10_clusters.pt',
    #    "FedRolex Random sizes": f'./checkpoints/MNIST/LeNet5/noniid/5x10/fedrolex.pt',
    #    "FedScavar C=10 Rolex" : f'./checkpoints/MNIST/LeNet5/noniid/5x10/fedscavar_10_clusters_rolex.pt'}
    #checkpoints_1  = {
    #    "FedRolex E=1"              : f'./checkpoints/MNIST/LeNet5/noniid/1x10/fedrolex.pt',
    #    "FedScavar Rolex C=10 E=1"  : f'./checkpoints/MNIST/LeNet5/noniid/1x10/fedscavar_10_clusters_rolex.pt',
    #    "FedRolex E=5"              : f'./checkpoints/MNIST/LeNet5/noniid/5x10/fedrolex.pt',
    #    "FedScavar Rolex C=10 E=5"  : f'./checkpoints/MNIST/LeNet5/noniid/5x10/fedscavar_10_clusters_rolex.pt',
    #    "FedRolex E=20"             : f'./checkpoints/MNIST/LeNet5/noniid/20x10/fedrolex.pt',
    #    "FedScavar Rolex C=10 E=20" : f'./checkpoints/MNIST/LeNet5/noniid/20x10/fedscavar_10_clusters_rolex.pt'
    #}
    model_datas_1 = {name: torch.load(path) for name, path in checkpoints_1.items()}

    checkpoints_2  = {
        "SCAFFOLD"             : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/20x10/scaffold.pt',
        "FedVARP"              : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/20x10/fedvarp.pt',
        "FedScavar C=100"      : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/20x10/fedscavar_100_clusters.pt'
    }
    #checkpoints_2 = {
    #    "ClusterFedVARP C=100" : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedvarp.pt',
    #    "ClusterFedVARP C=1"   : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/clustervarp_1_clusters.pt',
    #    "ClusterFedVARP C=10"  : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/clustervarp_10_clusters.pt',
    #    "FedScavar C=1"        : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedscavar_1_clusters.pt',
    #    "FedScavar C=10"       : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedscavar_10_clusters.pt',
    #    "FedScavar C=100"      : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedscavar_100_clusters.pt',
    #}
    #checkpoints_2  = {
    #    "FedAvg"               : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedavg.pt',
    #    "FedProx Mu=0.25"      : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedprox_0.25.pt',
    #    "SCAFFOLD"             : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/scaffold.pt',
    #    "FedVARP"              : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedvarp.pt',
    #    "FedScavar C=100"      : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedscavar_100_clusters.pt',
    #    "ClusterFedVARP C=1"   : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/clustervarp_1_clusters.pt',
    #    "ClusterFedVARP C=10"  : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/clustervarp_10_clusters.pt',
    #    "FedScavar C=1"        : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedscavar_1_clusters.pt',
    #    "FedScavar C=10"       : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedscavar_10_clusters.pt',
    #    "FedRolex Random sizes": f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedrolex.pt',
    #    "FedScavar C=10 Rolex" : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedscavar_10_clusters_rolex.pt'}
    #checkpoints_2  = {
    #    "FedRolex E=1"              : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/1x10/fedrolex.pt',
    #    "FedScavar Rolex C=10 E=1"  : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/1x10/fedscavar_10_clusters_rolex.pt',
    #    "FedRolex E=5"              : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedrolex.pt',
    #    "FedScavar Rolex C=10 E=5"  : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/5x10/fedscavar_10_clusters_rolex.pt',
    #    "FedRolex E=20"             : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/20x10/fedrolex.pt',
    #    "FedScavar Rolex C=10 E=20" : f'./checkpoints/MNIST/LeNet5/dirichlet/0.1/20x10/fedscavar_10_clusters_rolex.pt'
    #}
    model_datas_2 = {name: torch.load(path) for name, path in checkpoints_2.items()}

    checkpoints_3  = {
        "SCAFFOLD"             : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/20x10/scaffold.pt',
        "FedVARP"              : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/20x10/fedvarp.pt',
        "FedScavar C=100"      : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/20x10/fedscavar_100_clusters.pt'
    }
    #checkpoints_3 = {
    #    "ClusterFedVARP C=100" : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/5x10/fedvarp.pt',
    #    "ClusterFedVARP C=1"   : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/5x10/clustervarp_1_clusters.pt',
    #    "ClusterFedVARP C=10"  : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/5x10/clustervarp_10_clusters.pt',
    #    "FedScavar C=1"        : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/5x10/fedscavar_1_clusters.pt',
    #    "FedScavar C=10"       : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/5x10/fedscavar_10_clusters.pt',
    #    "FedScavar C=100"      : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/5x10/fedscavar_100_clusters.pt',
    #}
    #checkpoints_3  = {
    #    "FedAvg"               : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/5x10/fedavg.pt',
    #    "FedProx Mu=0.25"      : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/5x10/fedprox_0.25.pt',
    #    "SCAFFOLD"             : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/5x10/scaffold.pt',
    #    "FedVARP"              : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/5x10/fedvarp.pt',
    #    "FedScavar C=100"      : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/5x10/fedscavar_100_clusters.pt',
    #    "ClusterFedVARP C=1"   : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/5x10/clustervarp_1_clusters.pt',
    #    "ClusterFedVARP C=10"  : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/5x10/clustervarp_10_clusters.pt',
    #    "FedScavar C=1"        : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/5x10/fedscavar_1_clusters.pt',
    #    "FedScavar C=10"       : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/5x10/fedscavar_10_clusters.pt',
    #    "FedRolex Random sizes": f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/5x10/fedrolex.pt',
    #    "FedScavar C=10 Rolex" : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/5x10/fedscavar_10_clusters_rolex.pt'}
    #checkpoints_3  = {
    #    "FedRolex E=1"              : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/1x10/fedrolex.pt',
    #    "FedScavar Rolex C=10 E=1"  : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/1x10/fedscavar_10_clusters_rolex.pt',
    #    "FedRolex E=5"              : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/5x10/fedrolex.pt',
    #    "FedScavar Rolex C=10 E=5"  : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/5x10/fedscavar_10_clusters_rolex.pt',
    #    "FedRolex E=20"             : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/20x10/fedrolex.pt',
    #    "FedScavar Rolex C=10 E=20" : f'./checkpoints/MNIST/LeNet5/dirichlet/1.0/20x10/fedscavar_10_clusters_rolex.pt'
    #}

    model_datas_3 = {name: torch.load(path) for name, path in checkpoints_3.items()}

    checkpoints_4  = {
        "SCAFFOLD"             : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/20x10/scaffold.pt',
        "FedVARP"              : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/20x10/fedvarp.pt',
        "FedScavar C=100"      : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/20x10/fedscavar_100_clusters.pt'
    }
    #checkpoints_4 = {
    #    "ClusterFedVARP C=100" : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/fedvarp.pt',
    #    "ClusterFedVARP C=1"   : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/clustervarp_1_clusters.pt',
    #    "ClusterFedVARP C=10"  : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/clustervarp_10_clusters.pt',
    #    "FedScavar C=1"        : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/fedscavar_1_clusters.pt',
    #    "FedScavar C=10"       : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/fedscavar_10_clusters.pt',
    #    "FedScavar C=100"      : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/fedscavar_100_clusters.pt',
    #}
    #checkpoints_4  = {
    #    "FedAvg"               : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/fedavg.pt',
    #    "FedProx Mu=0.25"      : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/fedprox_0.25.pt',
    #    "SCAFFOLD"             : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/scaffold.pt',
    #    "FedVARP"              : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/fedvarp.pt',
    #    "FedScavar C=100"      : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/fedscavar_100_clusters.pt',
    #    "ClusterFedVARP C=1"   : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/clustervarp_1_clusters.pt',
    #    "ClusterFedVARP C=10"  : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/clustervarp_10_clusters.pt',
    #    "FedScavar C=1"        : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/fedscavar_1_clusters.pt',
    #    "FedScavar C=10"       : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/fedscavar_10_clusters.pt',
    #    "FedRolex Random sizes": f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/fedrolex.pt',
    #    "FedScavar C=10 Rolex" : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/fedscavar_10_clusters_rolex.pt'}
    #checkpoints_4  = {
    #    "FedRolex E=1"              : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/1x10/fedrolex.pt',
    #    "FedScavar Rolex C=10 E=1"  : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/1x10/fedscavar_10_clusters_rolex.pt',
    #    "FedRolex E=5"              : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/fedrolex.pt',
    #    "FedScavar Rolex C=10 E=5"  : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/5x10/fedscavar_10_clusters_rolex.pt',
    #    "FedRolex E=20"             : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/20x10/fedrolex.pt',
    #    "FedScavar Rolex C=10 E=20" : f'./checkpoints/MNIST/LeNet5/dirichlet/100.0/20x10/fedscavar_10_clusters_rolex.pt'
    #}
    model_datas_4 = {name: torch.load(path) for name, path in checkpoints_4.items()}

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, squeeze=True)

    ax1.set_title("accuracy FedAvg")
    ax2.set_title("accuracy FedProx")
    ax3.set_title("accuracy SCAFFOLD")
    ax4.set_title("accuracy FedVARP")

    #ax1.set_title("loss FedAvg")
    #ax2.set_title("loss FedProx")
    #ax3.set_title("loss SCAFFOLD")
    #ax4.set_title("loss FedVARP")

    #ax1.set_title("accuracy NONIID, steps=50")
    #ax2.set_title("accuracy alpha=0.1, steps=50")
    #ax3.set_title("accuracy alpha=1.0, steps=50")
    #ax4.set_title("accuracy alpha=100.0, steps=50")

    #ax1.set_title("accuracy NONIID")
    #ax2.set_title("accuracy alpha=0.1")
    #ax3.set_title("accuracy alpha=1.0")
    #ax4.set_title("accuracy alpha=100.0")

    #ax1.set_title("loss NONIID, steps=50")
    #ax2.set_title("loss alpha=0.1, steps=50")
    #ax3.set_title("loss alpha=1.0, steps=50")
    #ax4.set_title("loss alpha=100.0, steps=50")

    #ax1.set_title("loss NONIID")
    #ax2.set_title("loss alpha=0.1")
    #ax3.set_title("loss alpha=1.0")
    #ax4.set_title("loss alpha=100.0")

    for name, d in model_datas_1.items():
        plot_model2(ax1, d, name)
    for name, d in model_datas_2.items():
        plot_model2(ax2, d, name)
    for name, d in model_datas_3.items():
        plot_model2(ax3, d, name)
    for name, d in model_datas_4.items():
        plot_model2(ax4, d, name)
        
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    #plt.tight_layout()
    plt.show()

def algo_comparisons():
    checkpoints_clusterfedvarp_10  = {
        "ClusterFedVARP 10 Clusters C=0.01"  : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.01/clustervarp_10_clusters.pt',
        "ClusterFedVARP 10 Clusters C=0.1"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/clustervarp_10_clusters.pt',
        "ClusterFedVARP 10 Clusters C=0.5"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.5/clustervarp_10_clusters.pt',
    }
    model_datas_clusterfedvarp_10 = {name: torch.load(path) for name, path in checkpoints_clusterfedvarp_10.items()}

    checkpoints_fedavg  = {
        "FedAvg C=0.01"  : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.01/fedavg.pt',
        "FedAvg C=0.1"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/fedavg.pt',
        "FedAvg C=0.5"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.5/fedavg.pt',
    }
    model_datas_fedavg = {name: torch.load(path) for name, path in checkpoints_fedavg.items()}

    checkpoints_fedprox  = {
        "FedProx Mu=0.25 C=0.01"  : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.01/fedprox_0.25.pt',
        "FedProx Mu=0.25 C=0.1"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/fedprox_0.25.pt',
        "FedProx Mu=0.25 C=0.5"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.5/fedprox_0.25.pt'
    }
    model_datas_fedprox = {name: torch.load(path) for name, path in checkpoints_fedprox.items()}

    checkpoints_fedrolex  = {
        "FedRolex C=0.01"  : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.01/fedrolex.pt',
        "FedRolex C=0.1"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/fedrolex.pt',
        "FedRolex C=0.5"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.5/fedrolex.pt'
    }
    model_datas_fedrolex = {name: torch.load(path) for name, path in checkpoints_fedrolex.items()}

    checkpoints_fedscavar_10  = {
        "FedSCAVAR 10 Clusters C=0.01"  : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.01/fedscavar_10_clusters.pt',
        "FedSCAVAR 10 Clusters C=0.1"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/fedscavar_10_clusters.pt',
        "FedSCAVAR 10 Clusters C=0.5"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.5/fedscavar_10_clusters.pt'
    }
    model_datas_fedscavar_10 = {name: torch.load(path) for name, path in checkpoints_fedscavar_10.items()}

    checkpoints_fedscavar_10_rolex  = {
        "FedSCAVAR 10 Clusters Rolex C=0.01"  : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.01/fedscavar_10_clusters_rolex.pt',
        "FedSCAVAR 10 Clusters Rolex C=0.1"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/fedscavar_10_clusters_rolex.pt',
        "FedSCAVAR 10 Clusters Rolex C=0.5"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.5/fedscavar_10_clusters_rolex.pt'
    }
    model_datas_fedscavar_10_rolex = {name: torch.load(path) for name, path in checkpoints_fedscavar_10_rolex.items()}

    checkpoints_fedscavar_100  = {
        "FedSCAVAR C=0.01"  : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.01/fedscavar_100_clusters.pt',
        "FedSCAVAR C=0.1"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/fedscavar_100_clusters.pt',
        "FedSCAVAR C=0.5"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.5/fedscavar_100_clusters.pt'
    }
    model_datas_fedscavar_100 = {name: torch.load(path) for name, path in checkpoints_fedscavar_100.items()}

    checkpoints_fedvarp = {
        "FedVARP C=0.01"  : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.01/fedvarp.pt',
        "FedVARP C=0.1"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/fedvarp.pt',
        "FedVARP C=0.5"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.5/fedvarp.pt'
    }
    model_datas_fedvarp = {name: torch.load(path) for name, path in checkpoints_fedvarp.items()}

    checkpoints_scaffold  = {
        "SCAFFOLD C=0.01"  : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.01/scaffold.pt',
        "SCAFFOLD C=0.1"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/scaffold.pt',
        "SCAFFOLD C=0.5"   : f'./checkpoints/MNIST/LeNet5/noniid/1x10/c_0.5/scaffold.pt',
    }
    model_datas_scaffold = {name: torch.load(path) for name, path in checkpoints_scaffold.items()}

    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, squeeze=True)

    ax1.set_title("FedAvg accuracy")
    ax2.set_title("FedProx accuracy")
    ax3.set_title("FedAvg loss")
    ax4.set_title("FedProx loss")

    for name, d in model_datas_fedavg.items():
        plot_model2(ax1, d, name, True)
        plot_model2(ax3, d, name, False)
    for name, d in model_datas_fedprox.items():
        plot_model2(ax2, d, name, True)
        plot_model2(ax4, d, name, False)

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()  
    plt.show()


    fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2, 3, squeeze=True)

    ax1.set_title("FedVARP accuracy")
    ax2.set_title("SCAFFOLD accuracy")
    ax3.set_title("FedSCAVAR accuracy")
    ax4.set_title("FedVARP loss")
    ax5.set_title("SCAFFOLD loss")
    ax6.set_title("FedSCAVAR loss")

    for name, d in model_datas_fedvarp.items():
        plot_model2(ax1, d, name, True)
        plot_model2(ax4, d, name, False)
    for name, d in model_datas_scaffold.items():
        plot_model2(ax2, d, name, True)
        plot_model2(ax5, d, name, False)
    for name, d in model_datas_fedscavar_100.items():
        plot_model2(ax3, d, name, True)
        plot_model2(ax6, d, name, False)

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()  
    ax5.legend()
    ax6.legend()  
    plt.show()
    
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, squeeze=True)

    ax1.set_title("FedRolex accuracy")
    ax2.set_title("FedSCAVAR Rolex accuracy")
    ax3.set_title("FedRolex loss")
    ax4.set_title("FedSCAVAR Rolex loss")

    for name, d in model_datas_fedrolex.items():
        plot_model2(ax1, d, name, True)
        plot_model2(ax3, d, name, False)
    for name, d in model_datas_fedscavar_10_rolex.items():
        plot_model2(ax2, d, name, True)
        plot_model2(ax4, d, name, False)

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()  
    plt.show()
    
    fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, squeeze=True)

    ax1.set_title("ClusterFedVARP accuracy")
    ax2.set_title("FedSCAVAR accuracy")
    ax3.set_title("ClusterFedVARP loss")
    ax4.set_title("FedSCAVAR Rolex loss")

    for name, d in model_datas_clusterfedvarp_10.items():
        plot_model2(ax1, d, name, True)
        plot_model2(ax3, d, name, False)
    for name, d in model_datas_fedscavar_10.items():
        plot_model2(ax2, d, name, True)
        plot_model2(ax4, d, name, False)

    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()  
    plt.show()
    

single_comparsion()
#four_plots_comparison()
algo_comparisons()
