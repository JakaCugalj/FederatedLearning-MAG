Here's a sample README for your project:

# Federated Learning - MAG

This project implements several federated learning algorithms, including FedAvg, FedProx, SCAFFOLD, FedVARP, ClusterFedVARP and FedRolex.

## Requirements

- Python 3.10 or higher
- PyTorch
- torchvision
- NumPy
- Matplotlib
- DearPyGui

## Installation

To install the required libraries, run the following command:

```sh
conda create --name FederatedLearning-MAG-GPU pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -n FederatedLearning-MAG-GPU dearpygui matplotlib numpy
```

## Usage

```python

from src.servers.FederatedServerAvg import FederatedServerAvg, FederatedClientAvg
from src.models import LeNet5
import torchvision
import torchvision.transforms as transforms

transform           = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
local_training_data = torchvision.datasets.MNIST('./data', train=True , transform=transform, download=True)
local_test_data     = torchvision.datasets.MNIST('./data', train=False, transform=transform, download=True)

clients = []

datasets = src.utils.dataset_split2(dataset=local_training_data,
                                    number_of_clients=100,
                                    split_type="noniid",
                                    min_classes=2)

for k in range(100):
  client = FederatedClientAvg(id=f'client_{k}', self.datasets[k],
                              epochs=5, batch_size=0.1, lr=0.001,
                              momentum=0.0, device="cuda", parent_tag=None)
  self.clients.append(client)
        
server = FederatedServerAvg(model=LeNet5(), clients=clients, rounds=500, server_lr=1.0,
                            client_fraction=0.1, num_threads=5, post_round_callback=None, device="cuda")

server.fit()

```
