import math
from typing import Union
import torch
import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.nn as nn
from torch.utils.data import Subset

class ParameterDict(OrderedDict):
    def __init__(self,
                 source : Union[nn.Module, OrderedDict[str, torch.Tensor]] = None,
                 clone  : bool = False,
                 zero   : bool = False,
                 device : str  = None , **kwargs): 
        OrderedDict.__init__(self, **kwargs)  

        if isinstance(source, nn.Module):
            iter_items = source.named_parameters()
        elif isinstance(source, dict):
            iter_items = source.items()
        else:
            iter_items = None
            
        if iter_items != None:
            for name, param in iter_items:
                if zero:
                    self[name] = torch.zeros_like(param.data)
                elif clone:
                    self[name] = param.data.clone()
                else:
                    self[name] = param.data
                if device != None:
                    self[name] = self[name].to(device)
                    
    def euclidean_norm(self):
        return torch.cat([p.view(-1) for p in self.values()]).norm(2).item()
    
    def _check_keys(self, other):
        if self.keys() != other.keys():
            raise ValueError("Keys must match in both ParameterDicts.")
    
    def __str__(self):
        str = ""
        for key, value in self.items():
            str += f"{key} : {value}\n"
        return str
    
    def __add__(self, other):
        result = ParameterDict()
        if isinstance(other, (int, float, torch.Tensor)):
            for key, value in self.items():
                result[key] = value + other
        else:
            self._check_keys(other)
            for key, value in self.items():
                result[key] = value + other[key]
        return result
        
    def __iadd__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            for key, value in self.items():
                self[key] += other
        else:
            self._check_keys(other)
            for key, value in self.items():
                self[key] += other[key]
        return self
    
    def __sub__(self, other):
        result = ParameterDict()
        if isinstance(other, (int, float, torch.Tensor)):
            for key, value in self.items():
                result[key] = value - other
        else:
            self._check_keys(other)
            for key, value in self.items():
                result[key] = value - other[key]
        return result
        
    def __isub__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            for key, value in self.items():
                self[key] -= other
        else:
            self._check_keys(other)
            for key, value in self.items():
                self[key] -= other[key]
        return self
            
    def __mul__(self, other):
        result = ParameterDict()
        if isinstance(other, (int, float, torch.Tensor)):
            for key, value in self.items():
                result[key] = value * other
        else:
            self._check_keys(other)
            for key, value in self.items():
                result[key] = value * other[key]
        return result

    def __imul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            for key, value in self.items():
                self[key] *= other
        else:
            self._check_keys(other)
            for key, value in self.items():
                self[key] *= other[key]
        return self
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        result = ParameterDict()
        if isinstance(other, (int, float, torch.Tensor)):
            for key, value in self.items():
                result[key] = value / other
        else:
            self._check_keys(other)
            for key, value in self.items():
                result[key] = value / other[key]
        return result
        
    def __itruediv__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            for key, value in self.items():
                self[key] /= other
        else:
            self._check_keys(other)
            for key, value in self.items():
                self[key] /= other[key]
        return self
    
    
class EpochMetrics:
    def __init__(self):
        self.history_loss     = []
        self.history_accuracy = []
        self.reset()
        
    def __len__(self):
        return len(self.history_accuracy)
    
    def __str__(self):
        return f"EpochMetrics(acc={self.accuracy:04f}, loss={self.loss:04f}, total_loss={self.total_loss:04f}, total_samples={self.total_samples})"
    
    def reset(self):
        self.total_loss    = 0.0
        self.total_correct = 0
        self.total_samples = 0

    def update(self, outputs, targets, loss):
        _, predictions = torch.max(outputs, 1)
        self.total_loss    += loss.item() * targets.size(0)
        self.total_correct += (predictions == targets).sum().item()
        self.total_samples += targets.size(0)
    
    def archive_current(self):
        self.history_accuracy.append(self.accuracy)
        self.history_loss.append(self.loss)  
         
    @property
    def accuracy(self):
        if self.total_samples == 0:
            return 0.0
        return self.total_correct / self.total_samples

    @property
    def loss(self):
        if self.total_samples == 0:
            return 0.0
        return self.total_loss / self.total_samples



def dataset_histogram(dataset, subset_indices, ax=None):
    """
        Generates a pyplot histogram of labels
    Returns:
        (x, y): histogram values
    """
    unique_targets, counts = torch.unique(torch.Tensor(dataset.targets)[subset_indices], return_counts=True)

    x = torch.arange(len(dataset.classes))
    y = torch.zeros_like(x)
    
    y[unique_targets.long()] = counts

    if ax != None: ax.bar(x, y)

    return x, y

def dataset_split2(dataset, number_of_clients, split_type="iid", min_classes=2, plot=False, cncntrcn=0.1):
    random.seed(123)
    
    match split_type:   
        case "iid":
            rand_indices  = torch.randperm(len(dataset))        
            split_indices = torch.chunk(rand_indices, number_of_clients)     
                   
        case "random_ratios":
            rand_indices  = torch.randperm(len(dataset))    
            split_indices = [indices[:int(len(indices) * ratio)] 
                             for indices, ratio in zip(torch.chunk(rand_indices, number_of_clients), torch.rand(number_of_clients))]      
        
        case "dirichlet":
            K = number_of_clients
            def sample_with_mask(mask, ideal_samples_counts, concentration, need_adjustment=False):
                num_remaining_classes = int(mask.sum())
                
                # sample class selection probabilities based on Dirichlet distribution with concentration parameter (`diri_alpha`)
                selection_prob_raw = np.random.dirichlet(alpha=np.ones(num_remaining_classes) * concentration, size=1).squeeze()
                selection_prob = mask.copy()
                selection_prob[selection_prob == 1.] = selection_prob_raw
                selection_prob /= selection_prob.sum()

                # calculate per-class sample counts based on selection probabilities
                if need_adjustment: # if remaining samples are not enough, force adjusting sample sizes...
                    selected_counts = (selection_prob * ideal_samples_counts * np.random.uniform(low=0.0, high=1.0, size=len(selection_prob))).astype(int)
                else:
                    selected_counts = (selection_prob * ideal_samples_counts).astype(int)
                return selected_counts
            
            # get indices by class labels
            _, unique_inverse, unique_count = np.unique(dataset.targets, return_inverse=True, return_counts=True)
            class_indices = np.split(np.argsort(unique_inverse), np.cumsum(unique_count[:-1]))
            
            # make hashmap to track remaining samples per class
            class_samples_counts = dict(zip([i for i in range(len(dataset.classes))], [len(class_idx) for class_idx in class_indices]))
            
            # calculate ideal samples counts per client
            ideal_samples_counts = len(dataset.targets) // K
            if ideal_samples_counts < 1:
                raise Exception(f'[SIMULATE] Decrease the number of participating clients (`K` < {K})!')

            # assign divided shards to clients
            split_indices = []
            for k in range(K):
                # update mask according to the count of reamining samples per class
                # i.e., do NOT sample from class having no remaining samples
                remaining_mask = np.where(np.array(list(class_samples_counts.values())) > 0, 1., 0.)
                selected_counts = sample_with_mask(remaining_mask, ideal_samples_counts, cncntrcn)

                # check if enough samples exist per selected class
                expected_counts = np.subtract(np.array(list(class_samples_counts.values())), selected_counts)
                valid_mask = np.where(expected_counts < 0, 1., 0.)
                
                # if not, resample until enough samples are secured
                while sum(valid_mask) > 0:
                    # resample from other classes instead of currently selected ones
                    adjusted_mask = (remaining_mask.astype(bool) & (~valid_mask.astype(bool))).astype(float)
                    
                    # calculate again if enoush samples exist or not
                    selected_counts = sample_with_mask(adjusted_mask, ideal_samples_counts, cncntrcn, need_adjustment=True)    
                    expected_counts = np.subtract(np.array(list(class_samples_counts.values())), selected_counts)

                    # update mask for checking a termniation condition
                    valid_mask = np.where(expected_counts < 0, 1., 0.)
                    
                # assign shards in randomly selected classes to current client
                indices = []
                for it, counts in enumerate(selected_counts):
                    # get indices from the selected class
                    selected_indices = class_indices[it][:counts]
                    indices.extend(selected_indices)
                    
                    # update indices and statistics
                    class_indices[it] = class_indices[it][counts:]
                    class_samples_counts[it] -= counts
                else:
                    split_indices.append(indices)
            
        case "noniid":
            targets_tensor       = torch.Tensor(dataset.targets)
            sorted_indices       = torch.argsort(targets_tensor)
            sorted_targets       = targets_tensor[sorted_indices]
            num_targets          = len(torch.unique(targets_tensor))
            
            label_change_indices = torch.where(torch.diff(sorted_targets))[0] + 1
            shards_per_label     = number_of_clients * min_classes // num_targets     
            label_shards         = [label.chunk(shards_per_label) for label in sorted_indices.tensor_split(label_change_indices)]
            label_shard_counts   = torch.Tensor([len(label) for label in label_shards])
            
            split_indices = []   
            for c in range(number_of_clients):
                split_indices.append([])
                
                probabilities = torch.where(label_shard_counts > 0, 1., 0.)
                probabilities /= sum(probabilities) 
                    
                indices = torch.multinomial(probabilities, min_classes, False)
                
                for i in indices:       
                    label_ind = int(i)
                    shard_ind = int(label_shard_counts[i] - 1) 
                    split_indices[c] += label_shards[label_ind][shard_ind]
                    
                label_shard_counts[indices] -= 1
            
    if plot:
        num_cols = math.ceil(math.sqrt(number_of_clients))
        num_rows = math.ceil(number_of_clients / num_cols)

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(16, 8), squeeze=False)    

        for i, (indices, ax) in enumerate(zip(split_indices, axs.flatten())):
            try:
                if i == number_of_clients - 1:
                    ax.set_xticks(torch.arange(len(dataset.classes)))
                    ax.set_xticklabels(dataset.classes, rotation=90)
                    
                dataset_histogram(dataset, indices, ax)
            except StopIteration:
                break

        #if number_of_clients < num_rows * num_cols:
        #    for i in range(number_of_clients, num_rows * num_cols):
        #        fig.delaxes(axs[i])

        plt.tight_layout()
        plt.show()
        
    return [Subset(dataset, indices) for indices in split_indices]







def dataset_split(dataset, num_clients, classes_per_client=2, iid=True, plot=False):
    #assert len(dataset.classes) * classes_per_client >= num_clients, "Not enough classes, some clients will have no data"
    random.seed(123)
    
    if len(dataset.classes) > num_clients:
        print("warning, more classes than clients, clients will have more than desired classes_per_client")
        
    if iid:
        dataset_indices = torch.randperm(len(dataset))    
    else:
        dataset_indices = torch.argsort(torch.Tensor(dataset.targets)) 
   
    #find indices where labels change
    label_change_indices = torch.where(torch.diff(dataset.targets[dataset_indices]))[0] + 1
    
    #split dataset_indices for each label into separate tensor, and split that tensor into a list of num_parts
    label_chunks_indices = [list(torch.chunk(ind, (num_clients // len(dataset.classes)) * classes_per_client))
                            for ind in torch.tensor_split(dataset_indices, label_change_indices)]
    
    random.shuffle(label_chunks_indices)
    
    subsets_indices = [torch.Tensor().long() for i in range(num_clients)]
    client_idx      = 0
    for label_chunks in label_chunks_indices:
        for chunk in label_chunks:
            subsets_indices[client_idx] = torch.cat((subsets_indices[client_idx], chunk)) 
            client_idx = (client_idx + 1) % num_clients
                 
    if plot:
        for idx, subset_indices in enumerate(subsets_indices):
            ax1 = plt.subplot(max(1, int(num_clients / 5)+1), 5, idx+1)
            dataset_histogram(dataset, subset_indices, ax1)
        plt.show()  
           
    return [torch.utils.data.Subset(dataset, subset_indices) for subset_indices in subsets_indices]

def split_dataset2(dataset, num_clients, iid=False, num_parts=4, num_swaps=4, plot=False):
    if iid:
        dataset_indices = torch.randperm(len(dataset))    
    else:
        #sort
        dataset_indices = torch.argsort(dataset.targets)
        
        #find indices where labels change
        label_change_indices = torch.where(torch.diff(dataset.targets[dataset_indices]))[0] + 1
        
        #split dataset_indices for each label into separate tensor, and split that tensor into a list of num_parts
        label_chunk_indices = [list(torch.chunk(ind, num_parts))
                               for ind in torch.tensor_split(dataset_indices, label_change_indices)]
        
        #swap num_swaps random chunks with previous neighbour
        for i in range(len(label_chunk_indices)):
            prev_rand_ind = torch.randint(low=0, high=num_parts, size=(num_swaps,))
            next_rand_ind = torch.randint(low=0, high=num_parts, size=(num_swaps,))
            for s, (prev_idx, next_idx) in enumerate(zip(prev_rand_ind, next_rand_ind)):
                if s == num_swaps:
                    break
                if i >= 1:
                    label_chunk_indices[i-1][prev_idx], label_chunk_indices[i][prev_idx] = label_chunk_indices[i][prev_idx], label_chunk_indices[i-1][prev_idx]
                if i < len(label_change_indices) - 1:
                    label_chunk_indices[i+1][next_idx], label_chunk_indices[i][next_idx] = label_chunk_indices[i][next_idx], label_chunk_indices[i+1][next_idx]
                    
        #flatten indices into a 1D tensor
        dataset_indices = torch.cat([torch.cat(chunk, dim=0) for chunk in label_chunk_indices])
             
    #split indices for each client
    subsets_indices = torch.tensor_split(dataset_indices, num_clients)

    if plot:
        for idx, subset_indices in enumerate(subsets_indices):
            ax1 = plt.subplot(max(1, int(num_clients / 5)+1), 5, idx+1)
            dataset_histogram(dataset, subset_indices, ax1)
        plt.show()

    return [torch.utils.data.Subset(dataset, subset_indices) for subset_indices in subsets_indices]

def split_dataset(dataset, num_clients = 2, iid=False, imbalance=None):
    sorted_indices        = torch.argsort(dataset.targets) if not iid else torch.randperm(len(dataset))
    class_counters        = [0  for c in range(len(dataset.classes))]
    client_subset_indices = [[] for c in range(num_clients)]
    client_class_counter  = [[0  for i in range(len(dataset.classes))] for c in range(num_clients)]

    #for d in dataset.data[sorted_indices]:
    #    plt.imshow(d)
    #    plt.show()

    for input, label, idx in zip(dataset.data[sorted_indices], dataset.targets[sorted_indices], sorted_indices):
        client_subset_indices[class_counters[label.item()]].append(idx.item())

        client_class_counter[class_counters[label.item()]][label.item()] += 1

        if imbalance != None:
            if random.random() > imbalance[class_counters[label.item()]]:
                class_counters[label] = (class_counters[label] + 1) % num_clients

    m = max([max(l) for l in client_class_counter])
    for idx, l in enumerate(client_class_counter):
        ax1 = plt.subplot(5, int(num_clients/5)+1, idx+1)
        ax1.set_ylim([0, m])
        ax1.bar([i for i in range(len(dataset.classes))], l)
    plt.show()
    return [torch.utils.data.Subset(dataset, ind) for ind in client_subset_indices]

#datasets = split_dataset3(training_data, num_clients=10, classes_per_client=3, iid=True, plot=True)
#datasets = split_dataset2(training_data, 10, iid=False, num_parts=100, num_swaps=100, plot=True)
