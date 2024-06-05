import pandas as pd
import torch 
import torch.nn as nn
from typing import Callable, Dict, List, Tuple

from function import param, DiscretizeForward, SpikeCountLoss

hyper_params_: pd.DataFrame = pd.DataFrame({
    "num_neuron": [28 * 28, 800, 10],
    "tau_m": [0.07, 0.06, 0.05],
    "tau_s": [0.05, 0.04, 0.03],
    "resistance": [0.8, 1, 1.2],
    "threshold": [1.2, 1, 0.8],
    "reset": [0.5, 0.4, 0.3]
}, index=[0, 1, 2])




class LIF_Layer(nn.Module):
    def __init__(self, layer_idx: int, dim: int, prv_dim: int, prms: Dict, weight_init: Callable, device: str="cuda"):
        super(LIF_Layer, self).__init__()
        self.layer_idx = layer_idx
        self.dim = dim
        self.prms = prms
        self.weights = nn.Parameter(weight_init(prv_dim, dim).to(device))
    
    """
    Parameters:
    - prv_spikes: (batch_size, num_step, num_neuron) -> time 
    """
    def forward(self, prv_voltage: torch.Tensor, prv_spikes: torch.Tensor):
        DiscretizeForward.layer_idx = self.layer_idx
        return param(DiscretizeForward, self.prms).apply(prv_voltage, prv_spikes, self.weights)




class MNIST_Network(nn.Module):
    @staticmethod
    def _layer_config(params: pd.DataFrame, layer_idx: int) -> Dict:
        return {col + f"__{layer_idx}": params.loc[layer_idx, col] for col in params.columns}

    def __init__(self, num_layer: int, num_step: int, step_size: float, forward_type: int, hyper_params: pd.DataFrame=hyper_params_, device: str="cuda"):
        super(MNIST_Network, self).__init__()
        self.num_layer = num_layer
        self.num_step = num_step
        self.step_size = step_size
        self.forward_type = forward_type
        self.device = device

        init_configs = {"num_step": self.num_step, "step_size": self.step_size, "forward_type": self.forward_type, "device": self.device}
        global DiscretizeForward
        DiscretizeForward = param(DiscretizeForward, init_configs)
        
        assert num_layer == hyper_params.shape[0]
        self.layers = nn.ModuleList()

        prv_num_neuron = None
        for layer_idx in range(num_layer):
            layer_config: Dict = MNIST_Network._layer_config(hyper_params, layer_idx)
            num_neuron_ = f"num_neuron__{layer_idx}"
            num_neuron = layer_config[num_neuron_]
            del layer_config[num_neuron_]
            if prv_num_neuron is None: prv_num_neuron = num_neuron
            cur_layer: LIF_Layer = LIF_Layer(layer_idx, num_neuron, prv_num_neuron, layer_config, (lambda dim1, dim2: 2 * torch.rand((dim1, dim2)) - 1) if layer_idx != 0 else (lambda dim1, dim2: torch.ones((dim1, dim2))), self.device)
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
        
        for layer in self.layers:
            p_voltage, p_spikes = layer(p_voltage, p_spikes)
        
        return p_voltage, p_spikes
        
        
        
    

