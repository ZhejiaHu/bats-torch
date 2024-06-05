import threading
import torch
from typing import Callable, Dict, Tuple


def param(cls, configs: Dict):
    for prm_n, prm_v in configs.items(): setattr(cls, prm_n, prm_v)
    return cls    


class DiscretizeForward(torch.autograd.Function):
    
    du_dt_dict: Dict[int, torch.Tensor] = {}
    du_dt_dict_lock: threading.Lock = threading.Lock()


    @staticmethod
    def _device():
        return DiscretizeForward.device if hasattr(DiscretizeForward, "device") else "cpu"
    
    
    @staticmethod
    def _inv_du_dt(du_dt: torch.Tensor, scale: int=8):
        inv_du_dt: torch.Tensor = 1 / du_dt
        return torch.clamp(inv_du_dt, min=-scale, max=scale)


    """
    Parameters:
    - states: (batch_size, num_step, num_neuron, 2)
    - bias: (batch_size, num_step, num_neuron)

    Returns: (batch_size, num_step, num_neuron)
    """
    @staticmethod
    def _du_di(states: torch.Tensor, bias: torch.Tensor, resistance: float, tau_s: float, tau_m: float, reset: float, threshold: float) -> torch.Tensor:
        voltage, current = states[..., 0], states[..., 1]
        return tau_s / tau_m * (-voltage + resistance * current - reset * (voltage >= threshold)) / (-current + bias)
    
    
    """
    Parameters:
    - states: (batch_size, num_step, num_neuron, 2)
    
    Returns: (batch_size, num_step, num_neuron)
    """
    @staticmethod
    def _du_dt(states: torch.Tensor, resistance: float, tau_m: float, reset: float, threshold: float) -> torch.Tensor:
        voltage, current = states[..., 0], states[..., 1]
        return (-voltage + resistance * current - reset * (voltage >= threshold)) / tau_m
    

    """
    Notes:
    - Scalar time: partial derivative against one specific (spiking) time

    Parameters:
    - spikes: (batch_size, num_step, num_neuron)
    
    Returns: (batch_size, num_step, num_neuron)
    """
    @staticmethod
    def _pu_pt(spikes: torch.Tensor, cur_step: int, step_size: float, tau_m: float, reset: float) -> torch.Tensor:
        cur_time: float = cur_step * step_size 
        batch_size, num_step, num_neuron = spikes.shape
        times_: torch.Tensor = torch.arange(num_step).view(1, num_step, 1).expand(batch_size, num_step, num_neuron).to(DiscretizeForward._device())
        times: torch.Tensor = step_size * times_
        return (reset / (tau_m ** 2) * torch.exp(-(times - cur_time) / tau_m)) * spikes[:, cur_step].unsqueeze(1).expand(batch_size, num_step, num_neuron) * (times_ >= cur_step)
    

    """
    Parameters:
    - prv_spikes: (batch_size, num_step, num_neuron_prv)
    - weights: (num_neuron_prv, num_neuron_cur)
    
    Returns: (batch_size, num_step, num_neuron_cur)
    """
    @staticmethod
    def _pi_pt(prv_spikes: torch.Tensor, weights: torch.Tensor, cur_step: int, step_size: float, tau_s: float) -> torch.Tensor:
        cur_time: float = cur_step * step_size 
        batch_size, num_step, num_neuron = prv_spikes.shape
        times_: torch.Tensor = torch.arange(num_step).view(1, num_step, 1).expand(batch_size, num_step, num_neuron).to(DiscretizeForward._device())
        times: torch.Tensor = step_size * times_ 
        pi_pt_: torch.Tensor = 1 / (tau_s ** 2) * torch.exp(-(times - cur_time) / tau_s) * prv_spikes[:, cur_step].unsqueeze(1).expand(batch_size, num_step, num_neuron) * (times_ >= cur_step)
        return torch.einsum("bnx,xy->bny", pi_pt_, weights)


    @staticmethod 
    def _pi_pt_(prv_spikes: torch.Tensor, weights: torch.Tensor, cur_step: int, step_size: float, tau_s: float) -> torch.Tensor:
        cur_time: float = cur_step * step_size 
        batch_size, num_step, num_neuron = prv_spikes.shape
        times_: torch.Tensor = torch.arange(num_step).view(1, num_step, 1).expand(batch_size, num_step, num_neuron).to(DiscretizeForward._device())
        times: torch.Tensor = step_size * times_ 
        pi_pt_: torch.Tensor = 1 / (tau_s ** 2) * torch.exp(-(times - cur_time) / tau_s) * prv_spikes[:, cur_step].unsqueeze(1).expand(batch_size, num_step, num_neuron) * (times_ >= cur_step)
        return torch.einsum("bnx,xy->bnyx", pi_pt_, weights)

    """
    Parameters:
    - prv_spikes: (batch_size, num_step, num_neuron_prv)
    - weights: (num_neuron_prv, num_neuron_cur)

    Returns: (batch_size, num_step, num_neuron_prv, num_neuron_cur)

    Variables:
    - deriv_: (batch_size, num_time_all_times, num_time_spikes_times, num_neuron_prv)
    - deriv: (batch_sizem, num_time, num_neuron_prv)
    """
    @staticmethod 
    def _pu_pw(prv_spikes: torch.Tensor, weights: torch.Tensor, step_size: float, cur_resistance: float, cur_tau_m: float, cur_tau_s: float) -> torch.Tensor:
        num_neuron_prv, num_neuron_cur = weights.shape
        batch_size, num_step, _ = prv_spikes.shape 
        times: torch.Tensor = torch.arange(num_step).view(1, num_step, 1, 1).expand(batch_size, num_step, num_step, num_neuron_prv).to(DiscretizeForward._device())
        prv_spikes_: torch.Tensor = torch.where(prv_spikes.view(batch_size, 1, num_step, num_neuron_prv).expand(batch_size, num_step, num_step, num_neuron_prv) == 1, torch.arange(num_step).view(1, 1, num_step, 1).expand(batch_size, num_step, num_step, num_neuron_prv).to(DiscretizeForward._device()), num_step)   
        deriv_: torch.Tensor = (cur_resistance / (cur_tau_m - cur_tau_s) * (torch.exp(- (times * step_size - prv_spikes_ * step_size) / cur_tau_m) - torch.exp(- (times * step_size - prv_spikes_ * step_size) / cur_tau_s))) * (times >= prv_spikes_)
        deriv: torch.Tensor = torch.sum(deriv_, dim=2)
        return deriv.unsqueeze(-1).repeat(1, 1, 1, num_neuron_cur)


    """
    Parameters:
    - prv_spikes: (batch_size, num_step, num_neuron_prv)
    - weights: (num_neuron_prv, num_neuron_cur)
    - grad_u: (batch_size, num_step + 1, num_neuron_cur)

    Returns: (num_neuron_prv, num_neuron_cur)

    Variables:
    - pu_pw_: (batch_size, num_time (+ 1), num_neuron_prv, num_neuron_cur)
    """
    @staticmethod
    def _pe_pw(prv_spikes: torch.Tensor, weights: torch.Tensor, grad_u: torch.Tensor, step_size: float, cur_resistance: float, cur_tau_m: float, cur_tau_s: float) -> torch.Tensor:
        num_neuron_prv, num_neuron_cur = weights.shape
        batch_size, num_step, _ = prv_spikes.shape 
        pu_pw_: torch.Tensor = DiscretizeForward._pu_pw(prv_spikes, weights, step_size, cur_resistance, cur_tau_m, cur_tau_s)
        pu_pw_ = torch.cat((pu_pw_, torch.zeros((batch_size, 1, num_neuron_prv, num_neuron_cur), device=DiscretizeForward._device())), dim = 1)
        pe_pu_: torch.Tensor = torch.einsum("bnxy,bny->bxy", pu_pw_, grad_u)
        return torch.sum(pe_pu_, dim=0)
    
    
    
    """
    Parameters:
    - prv_spikes: (batch_size, num_step, num_neuron_prv)
    - weights: (num_neuron_prv, num_neuron_cur)

    Returns: (batch_size, num_step + 1, num_step + 1, num_neuron_cur, num_neuron_prv)

    Variables:
    - last_spikes: (batch_size, num_step + 1 (current time), num_neuron_prv)
    """    
    @staticmethod 
    def _pi_pu(batch_size: int, num_step: int, num_neuron_cur: int, num_neuron_prv: int, step_size: float, cur_tau_s: float, prv_tau_m: float, prv_reset: float,
               prv_spikes: torch.Tensor, weights: torch.Tensor, prv_du_dt: torch.Tensor) -> torch.Tensor:
        pi_pu: torch.Tensor = torch.zeros((batch_size, num_step + 1, num_step + 1, num_neuron_cur, num_neuron_prv), device=DiscretizeForward._device())
        last_spikes = torch.arange(num_step + 1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(batch_size, 1, num_neuron_cur, num_neuron_prv).to(DiscretizeForward._device())

        for prv_step in range(num_step - 1, -1, -1):
            pi_pt_ = DiscretizeForward._pi_pt_(prv_spikes, weights, prv_step, step_size, cur_tau_s)                                                                                                                     
            du_dt_ = prv_du_dt[:, prv_step]                                                                                                                                                                            
            pi_pu_ = torch.einsum("bnxy,by->bnxy", torch.cat((pi_pt_, torch.zeros((batch_size, 1, num_neuron_cur, num_neuron_prv), device=DiscretizeForward._device())), 1), DiscretizeForward._inv_du_dt(du_dt_))                                                                                      
            pu_pt__: torch.Tensor = torch.cat((DiscretizeForward._pu_pt(prv_spikes, prv_step, step_size, prv_tau_m, prv_reset), torch.zeros((batch_size, 1, num_neuron_prv), device=DiscretizeForward._device())), dim=1)                                                                                              
            pu_pt_: torch.Tensor = torch.gather(pu_pt__.unsqueeze(2).repeat(1, 1, num_neuron_cur, 1), 1, last_spikes)                     
            pu_pu_: torch.Tensor = torch.einsum("bnxy,by->bnxy", pu_pt_, DiscretizeForward._inv_du_dt(du_dt_))                                                                                                                                                               
            pi_pu__: torch.Tensor = torch.gather(pi_pu, 2, last_spikes.view(batch_size, num_step + 1, 1, num_neuron_cur, num_neuron_prv)).squeeze(2)                       
            pi_pu[:, :, prv_step] = pi_pu_ + pi_pu__ * pu_pu_
            last_spikes = torch.where(prv_spikes[:, prv_step].view(batch_size, 1, 1, num_neuron_prv).expand(batch_size, num_step + 1, num_neuron_cur, num_neuron_prv) == 1, torch.min(last_spikes, torch.tensor(prv_step)), last_spikes)
        return pi_pu
    





    """
    Parameters:
    -- prv_voltage: (batch_size, num_step, num_neuron_prv)
    -- prv_spikes: (batch_size, num_step, num_neuron_prv)
    -- weights: (num_neuron_prv, num_neuron_cur)

    Returns:
    -- cur_voltage: (batch_size, num_step, num_neuron_cur)
    -- cur_spikes: (batch_size, num_step, num_neuron_cur)

    Variables:
    -- bias: (batch_size, num_step, num_neuron_cur)
    -- gen_spike: (batch_size, num_neuron)
    """
    @staticmethod
    def forward(ctx, prv_voltage: torch.Tensor, prv_spikes: torch.Tensor, weights: torch.Tensor):   
        batch_size, num_step, num_neuron_prv = prv_voltage.shape
        _, num_neuron_cur = weights.shape

        bias: torch.Tensor = torch.einsum("bnx,xy->bny", prv_spikes, weights)
        step_size: float = DiscretizeForward.step_size
        forward_type: int = DiscretizeForward.forward_type
        layer_idx: int = DiscretizeForward.layer_idx
        
        tau_m: float = getattr(DiscretizeForward, f"tau_m__{layer_idx}")
        tau_s: float = getattr(DiscretizeForward, f"tau_s__{layer_idx}")
        resistance: float = getattr(DiscretizeForward, f"resistance__{layer_idx}")
        threshold: float = getattr(DiscretizeForward, f"threshold__{layer_idx}")
        reset: float = getattr(DiscretizeForward, f"reset__{layer_idx}")

        cur_state: torch.Tensor = torch.zeros((batch_size, num_step, num_neuron_cur, 2), device=DiscretizeForward._device(), requires_grad=True)
        cur_spikes: torch.Tensor = torch.zeros((batch_size, num_step, num_neuron_cur), device=DiscretizeForward._device(), requires_grad=True)
        cur_state[:, 0] = torch.rand((batch_size, num_neuron_cur, 2))
        if forward_type == 1: # Forward Euler
            mul_mat: torch.Tensor = torch.tensor([[1 - step_size / tau_m, resistance * step_size / tau_m], [0, 1 - step_size / tau_s]], device=DiscretizeForward._device(), dtype=torch.float32)
            for i in range(1, num_step):
                intercept: torch.Tensor = torch.zeros((batch_size, num_neuron_cur, 2), device=DiscretizeForward._device())
                intercept[:, :, 1] = step_size / tau_s * bias[:, i]
                cur_state[:, i] = torch.einsum("xy,bny->bnx", mul_mat, cur_state[:, i-1]) + intercept
                gen_spike = cur_state[:, i, :, 0] >= threshold
                cur_spikes[:, i] = gen_spike + 0
                cur_state[:, i, :, 0] -= gen_spike * reset

        cur_du_di = DiscretizeForward._du_di(cur_state, bias, resistance, tau_s, tau_m, reset, threshold)
        cur_du_dt = DiscretizeForward._du_dt(cur_state, resistance, tau_m, reset, threshold)
        ctx.layer_idx = layer_idx
        ctx.save_for_backward(prv_voltage, prv_spikes, cur_state, cur_spikes, weights, cur_du_di)
        with DiscretizeForward.du_dt_dict_lock:
            DiscretizeForward.du_dt_dict.setdefault(layer_idx, cur_du_dt)
        return cur_state[..., 0], cur_spikes


    """
    Parameters:
    -- grad_voltage_cur: (batch_size, num_step, num_neuron_cur)
    
    Variables:
    -- prv_du_dt: (batch_size, num_step, num_neuron_prv)
    """
    @staticmethod 
    def backward(ctx, cur_grad_voltage_: torch.Tensor, cur_grad_spikes: torch.Tensor):
        layer_idx = ctx.layer_idx
        prv_voltage, prv_spikes, cur_state, cur_spikes, weights, cur_du_di = ctx.saved_tensors      
        if layer_idx == 0: return None, None, None

        batch_size, num_step, num_neuron_cur = cur_grad_voltage_.shape
        num_neuron_prv = prv_spikes.shape[-1]

        step_size: float        = DiscretizeForward.step_size
        cur_tau_m: float        = getattr(DiscretizeForward, f"tau_m__{layer_idx}")
        cur_tau_s: float        = getattr(DiscretizeForward, f"tau_s__{layer_idx}")
        cur_resistance: float   = getattr(DiscretizeForward, f"resistance__{layer_idx}")
        prv_tau_m: float        = getattr(DiscretizeForward, f"tau_m__{layer_idx - 1}")
        prv_reset: float        = getattr(DiscretizeForward, f"reset__{layer_idx - 1}")


        with DiscretizeForward.du_dt_dict_lock:
            prv_du_dt: torch.Tensor = DiscretizeForward.du_dt_dict[layer_idx - 1]
        
        grad_current_cur: torch.Tensor = torch.cat((cur_grad_voltage_ * cur_du_di, torch.zeros((batch_size, 1, num_neuron_cur), device=DiscretizeForward._device())), dim=1)
        cur_grad_voltage: torch.Tensor = torch.cat((cur_grad_voltage_, torch.zeros((batch_size, 1, num_neuron_cur), device=DiscretizeForward._device())), dim=1)                                                                                                                                                   # (batch_size, num_step + 1, num_neuron_cur)
        grad_weights: torch.Tensor = DiscretizeForward._pe_pw(prv_spikes, weights, cur_grad_voltage, step_size, cur_resistance, cur_tau_m, cur_tau_s)
        
        pi_pu: torch.Tensor = DiscretizeForward._pi_pu(batch_size, num_step, num_neuron_cur, num_neuron_prv, step_size, cur_tau_s, prv_tau_m, prv_reset, 
                                                       prv_spikes, weights, prv_du_dt)
            

        grad_voltage_prv = torch.einsum("bmnxy,bmx->bny", pi_pu, grad_current_cur)
        assert grad_voltage_prv.shape == (batch_size, num_step + 1, num_neuron_prv)                                                                                                                                      # (batch_size, num_step + 1, num_neuron_prv)
        return grad_voltage_prv[:, :-1], None, grad_weights


class SpikeCountLoss(torch.autograd.Function):
    """
    Notes:
    - num_class == num_neuron
    Parameters:
    - output_spikes: (batch_size, num_step, num_class) 
    - target_spikes: (batch_size, num_class) 
    """
    @staticmethod
    def forward(ctx, output_voltage: torch.Tensor, output_spikes: torch.Tensor, target_spikes: torch.Tensor) -> float:
        batch_size, num_step, num_class = output_spikes.shape
        output_spikes_cnt: torch.Tensor = torch.sum(output_spikes, dim=1)
        ctx.save_for_backward(output_spikes, target_spikes)
        return torch.sum(torch.abs(output_spikes_cnt - target_spikes), dim=(0, 1)) / batch_size
    

    """
    Returns: (batch_size, num_step + 1, num_class)

    Variables:
    - diff_cnt: (batch_size, num_class)
    """
    @staticmethod
    def backward(ctx, grad_outputs) -> torch.Tensor:
        output_spikes, target_spikes = ctx.saved_tensors
        batch_size, num_step, num_class = output_spikes.shape
        output_spikes_cnt: torch.Tensor = torch.sum(output_spikes, dim=1)
        diff_cnt = torch.abs(output_spikes_cnt - target_spikes) / num_step
        return diff_cnt.unsqueeze(1).repeat(1, num_step, 1), None, None
    

    """
    Parameters:
    - output_spikes: (batch_size, num_step, num_neuron)
    - labels: (batch_size,)
    """
    @staticmethod
    def accuracy(output_spikes: torch.Tensor, labels: torch.Tensor) -> int:
        with torch.no_grad():
            predictions: torch.Tensor = torch.argmax(output_spikes.sum(dim=1), dim=-1)
            return torch.count_nonzero(predictions == labels)
        




        

        



