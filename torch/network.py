import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
from torch.optim import Adam, Optimizer
from typing import Callable, Dict, List, Tuple

from function import param, DiscretizeForward, SpikeCountLoss


class LIF_Layer(nn.Module):
    def __init__(self, layer_idx: int, dim: int, prv_dim: int, prms: Dict, weight_init: Callable, device: str="cuda", kaiming: bool=True):
        super(LIF_Layer, self).__init__()
        self.layer_idx = layer_idx
        self.dim = dim
        self.prms = prms
        self.weights = nn.Parameter(weight_init(prv_dim, dim).to(device))
        if kaiming: nn.init.kaiming_normal_(self.weights)
    
    """
    Parameters:
    - prv_spikes: (batch_size, num_step, num_neuron) -> time 
    """
    def forward(self, prv_voltage: torch.Tensor, prv_spikes: torch.Tensor):
        DiscretizeForward.layer_idx = self.layer_idx
        return param(DiscretizeForward, self.prms).apply(prv_voltage, prv_spikes, self.weights)




class MNIST_Network(nn.Module):
    @staticmethod
    def _layer_config(params: pd.DataFrame, layer_idx: int, add_idx: bool=True) -> Dict:
        return {col + (f"__{layer_idx}" if add_idx else ""): params.loc[layer_idx, col] for col in params.columns}

    def __init__(self, num_layer: int, num_step: int, step_size: float, forward_type: int, hyper_params: pd.DataFrame, device: str="cuda"):
        super(MNIST_Network, self).__init__()
        self.num_layer = num_layer
        self.num_step = num_step
        self.step_size = step_size
        self.forward_type = forward_type
        self.device = device

        init_configs = {"num_step": self.num_step, "step_size": self.step_size, "forward_type": self.forward_type, "device": self.device}
        global DiscretizeForward, SpikeCountLoss
        DiscretizeForward = param(DiscretizeForward, init_configs)
        
        assert num_layer == hyper_params.shape[0]
        self.layers = nn.ModuleList()

        prv_num_neuron = None
        for layer_idx in range(num_layer):
            layer_config: Dict = MNIST_Network._layer_config(hyper_params, layer_idx)
            if layer_idx == num_layer - 1:
                SpikeCountLoss = param(SpikeCountLoss, MNIST_Network._layer_config(hyper_params, layer_idx, add_idx=False))
                setattr(SpikeCountLoss, "step_size", step_size)
                setattr(SpikeCountLoss, "device", device)
            num_neuron_ = f"num_neuron__{layer_idx}"
            num_neuron = layer_config[num_neuron_]
            del layer_config[num_neuron_]
            if prv_num_neuron is None: prv_num_neuron = num_neuron
            cur_layer: LIF_Layer = LIF_Layer(layer_idx, num_neuron, prv_num_neuron, layer_config, (lambda dim1, dim2: 2 * torch.rand((dim1, dim2)) - 1), self.device)
            self.layers.append(cur_layer)
            prv_num_neuron = num_neuron


    """
    Parameters:
    - input_spikes_: (batch_size, num_step, num_neuron)
    """
    def forward(self, input_spikes_: torch.Tensor):
        assert input_spikes_.shape[1] == self.num_step
        p_spikes: torch.Tensor = input_spikes_.to(self.device).type(torch.float32)
        p_voltage: torch.Tensor = torch.ones_like(p_spikes).to(self.device)
        p_current: torch.Tensor = None

        for layer in self.layers:
            p_voltage, p_current, p_spikes = layer(p_voltage, p_spikes)
        
        return p_voltage, p_current, p_spikes
    
    def get_weights(self, idx: int):
        return self.layers[idx].weights.data
        
permutate_weight: Callable = lambda weights: weights[np.lexsort(weights.transpose(), axis=0)]


def convert_to_linear_ann(network: MNIST_Network) -> Tuple[nn.Module, Optimizer]:
    linear_ann: nn.Module = nn.Sequential(nn.Flatten())
    for i in range(network.num_layer):
        linear_ann.append(nn.Linear(network.get_weights(i).shape[0], network.get_weights(i).shape[1]))
        if i < network.num_layer - 1: linear_ann.append(nn.ReLU())
    optimizer = Adam(linear_ann.parameters(), lr=0.001)
    return linear_ann, optimizer 
    
    
