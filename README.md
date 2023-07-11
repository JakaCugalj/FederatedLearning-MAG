# Federated Learning - MAG

This project implements several federated learning algorithms, including FedAvg, FedProx, SCAFFOLD, FedVARP, ClusterFedVARP and FedRolex. It also provides an implementation of all algorithms combined called FedSCAVAR.

## Requirements

- Python 3.10 or higher
- PyTorch
- torchvision
- NumPy
- Matplotlib
- DearPyGui

## Installation

To install the required libraries, run the following command:

### CUDA
```sh
conda create --name FederatedLearning-MAG-GPU pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -n FederatedLearning-MAG-GPU dearpygui matplotlib numpy
```

### CPU
```sh
conda create --name FederatedLearning-MAG-GPU pytorch torchvision torchaudio cpuonly -c pytorch
conda install -n FederatedLearning-MAG-GPU dearpygui matplotlib numpy
```

## Usage
### UI
```python
conda activate FederatedLearning-MAG-GPU
python main.py
```

![image](https://github.com/JakaCugalj/FederatedLearning-MAG/assets/33024213/3fa69b84-0939-4ba9-adc9-345d4884ff87)

### Code example usage of FedAvg

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

## References

- McMahan, H. Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." arXiv:1602.05629 (2016). [Link](https://arxiv.org/abs/1602.05629)

- Li, Tian, et al. "Federated optimization in heterogeneous networks." arXiv:1812.06127 (2018). [Link](https://arxiv.org/abs/1812.06127)

- Karimireddy, Sai Praneeth Reddy, et al. "SCAFFOLD: Stochastic controlled averaging for on-device federated learning." arXiv:1910.06378 (2019). [Link](https://arxiv.org/abs/1910.06378)

- Jhunjhunwala, Divyansh, et al. "FedVARP: Tackling the Variance Due to Partial Client Participation in Federated Learning" arXiv:2207.14130 (2022). [Link](https://arxiv.org/abs/2207.14130)

- Alam, Samiul, et al. "FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction" arXiv:2207.14130 (2022). [Link](https://arxiv.org/abs/2212.01548)
