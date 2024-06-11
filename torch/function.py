import threading
import torch
from typing import Callable, Dict, Tuple

BIG_NUMBER = 1e6


def param(cls, configs: Dict):
    for prm_n, prm_v in configs.items(): setattr(cls, prm_n, prm_v)
    return cls    


class DiscretizeForward(torch.autograd.Function):
    
    du_dt_dict: Dict[int, torch.Tensor] = {}
    du_dt_dict_lock: threading.Lock = threading.Lock()



    @staticmethod
    def _device():
        return DiscretizeForward.device if hasattr(DiscretizeForward, "device") else "cpu"
    
    """
    Assert non NaN values and abnormal large values.
    """
    @staticmethod
    def _assert(name: str, grad: torch.Tensor):
        assert not torch.any(torch.isnan(grad)) and not torch.all(grad == 0) and not torch.any(grad > BIG_NUMBER), f"Gradient {name} has {torch.count_nonzero(torch.isnan(grad)) / torch.numel(grad) * 100}% NaN values, {torch.count_nonzero(grad == 0) / torch.numel(grad) * 100}% zero values, and {torch.count_nonzero(grad > BIG_NUMBER) / torch.numel(grad) * 100}% values exceeding {BIG_NUMBER}"

    
    
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
    def _du_di(states: torch.Tensor, bias: torch.Tensor, resistance: float, tau_s: float, tau_m: float, reset: float, threshold: float, clamp=True, clamp_val=10) -> torch.Tensor:
        voltage, current = states[..., 0], states[..., 1]
        tmp = tau_s / tau_m * (-voltage + resistance * current - reset * (voltage >= threshold)) / (-current + bias)
        if clamp: tmp = torch.clamp(tmp, min=-clamp_val, max=clamp_val)
        return tmp 
    

    
    """
    Use discretized version of du_di, which is simply a constant irregardless of everything.
    """
    @staticmethod 
    def _du_di_discretize(states: torch.Tensor, step_size: float, resistance: float, tau_m: float) -> torch.Tensor:
        return torch.full(states.shape[:-1], resistance * step_size / (step_size + tau_m), device=DiscretizeForward._device())

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
        return torch.mean(pe_pu_, dim=0)
    


    """
    Parameters:
    - recur_di: (batch_size, num_step, num_neuron_cur)
    - weights: (num_neuron_pre, num_neuron_cur)
    - prv_spikes: (batch_size, num_step, num_neuron_prv)
    
    """
    @staticmethod
    def _pe_pw_i(recur_di: torch.Tensor, weights: torch.Tensor, prv_spikes: torch.Tensor, step_size: float, tau_s: float) -> torch.Tensor:
        coeff: float = step_size / (tau_s + step_size)
        pi_pw_: torch.Tensor = coeff * torch.einsum("bnx,bny->bnxy", prv_spikes, recur_di)
        return torch.mean(pi_pw_, dim=(0, 1))
    
    
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
        cur_steps = torch.arange(num_step + 1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(batch_size, 1, num_neuron_cur, num_neuron_prv).to(DiscretizeForward._device())
        for prv_step in range(num_step - 1, -1, -1):
            pi_pt_ = DiscretizeForward._pi_pt_(prv_spikes, weights, prv_step, step_size, cur_tau_s)                                                                                                                     
            du_dt_ = prv_du_dt[:, prv_step]                                                                                                                                                                            
            pi_pu_ = torch.einsum("bnxy,by->bnxy", torch.cat((pi_pt_, torch.zeros((batch_size, 1, num_neuron_cur, num_neuron_prv), device=DiscretizeForward._device())), 1), DiscretizeForward._inv_du_dt(du_dt_))                                                                                      
            pu_pt__: torch.Tensor = torch.cat((DiscretizeForward._pu_pt(prv_spikes, prv_step, step_size, prv_tau_m, prv_reset), torch.zeros((batch_size, 1, num_neuron_prv), device=DiscretizeForward._device())), dim=1)                                                                                              
            pu_pt_: torch.Tensor = torch.gather(pu_pt__.unsqueeze(2).repeat(1, 1, num_neuron_cur, 1), 1, last_spikes)                     
            pu_pu_: torch.Tensor = torch.einsum("bnxy,by->bnxy", pu_pt_, DiscretizeForward._inv_du_dt(du_dt_))                                                                                                                                                               
            pi_pu__: torch.Tensor = torch.gather(pi_pu, 2, last_spikes.view(batch_size, num_step + 1, 1, num_neuron_cur, num_neuron_prv)).squeeze(2)                       
            pi_pu[:, :, prv_step] = pi_pu_ + torch.where(torch.logical_and(cur_steps != prv_step, prv_step != last_spikes),  pi_pu__ * pu_pu_, 0)
            last_spikes = torch.where(prv_spikes[:, prv_step].view(batch_size, 1, 1, num_neuron_prv).expand(batch_size, num_step + 1, num_neuron_cur, num_neuron_prv) == 1, torch.min(last_spikes, torch.tensor(prv_step)), last_spikes)
        return pi_pu
    

    """
    Parameters:
    - grad_voltage_cur_: (batch_size, num_step, num_neuron_cur)

    Returns: (batch_size, num_step, num_neuron_cur)
    """
    @staticmethod 
    def _recur_di(states_cur: torch.Tensor, grad_voltage_cur_: torch.Tensor, bias: torch.Tensor, step_size: float, resistance: float, tau_m: float, tau_s: float, reset: float, threshold: float, discretize: bool=True) -> torch.Tensor:
        batch_size, num_step, num_neuron_cur = grad_voltage_cur_.shape
        recur_di: torch.Tensor = torch.zeros_like(grad_voltage_cur_, device=DiscretizeForward._device())
        du_di: torch.Tensor = DiscretizeForward._du_di_discretize(states_cur, step_size, resistance, tau_m) if discretize else DiscretizeForward._du_di(states_cur, bias, resistance, tau_s, tau_m, reset, threshold)
        recur_di[:, -1] = grad_voltage_cur_[:, -1] * du_di[:, -1]
        for t in range(num_step - 2, -1, -1):
            recur_di[:, t] = grad_voltage_cur_[:, t] * du_di[:, t] + recur_di[:, t + 1] * tau_s / (tau_s + step_size)
        return recur_di  


    """
    Parameters:
    - recur_di: (batch_size, num_step, num_neuron_cur)
    - pi_pu: (batch_size, num_step (cur), num_step (prv), num_neuron_cur, num_neuron_prv)
    

    Returns:
    - recur_du: (batch_size, num_step, num_neuron_prv)
    """
    @staticmethod
    def _recur_du(recur_di: torch.Tensor, pi_pu: torch.Tensor, step_size: float, prv_tau_m: float) -> torch.Tensor:
        batch_size, num_step, _, num_neuron_cur, num_neuron_prv = pi_pu.shape 
        recur_du: torch.Tensor = torch.zeros((batch_size, num_step, num_neuron_prv), device=DiscretizeForward._device())
        recur_du[:, -1] = torch.einsum("bxy,bx->by", pi_pu[:, -1, -1], recur_di[:, -1])
        for t_prv in range(num_step - 2, -1, -1):
            recur_du[:, t_prv] = prv_tau_m / (prv_tau_m + step_size) * recur_du[:, t_prv + 1]
            for t_cur in range(t_prv, num_step):
                recur_du[:, t_prv] += torch.einsum("bxy,bx->by", pi_pu[:, t_cur, t_prv], recur_di[:, t_cur]) 
        return recur_du

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
    -- intercept: (batch_size, num_neuron, 2)
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
        cur_state[:, 0] = torch.ones((batch_size, num_neuron_cur, 2))
        if forward_type == 1: # Bacward Euler
            mul_mat: torch.Tensor = torch.linalg.inv(torch.tensor([[1 + step_size / tau_m, - resistance * step_size / tau_m], [0, 1 + step_size / tau_s]], device=DiscretizeForward._device(), dtype=torch.float32))
            for i in range(1, num_step):
                intercept: torch.Tensor = torch.zeros((batch_size, num_neuron_cur, 2), device=DiscretizeForward._device())
                intercept[:, :, 1] = bias[:, i] / tau_s
                cur_state[:, i] = torch.einsum("xy,bny->bnx", mul_mat, cur_state[:, i-1] + intercept)
                gen_spike = cur_state[:, i, :, 0] >= threshold
                cur_spikes[:, i] = gen_spike + 0
                cur_state[:, i, :, 0] -= gen_spike * reset

        #cur_du_di = DiscretizeForward._du_di(cur_state, bias, resistance, tau_s, tau_m, reset, threshold)
        cur_du_di = DiscretizeForward._du_di_discretize(cur_state, step_size, resistance, tau_m)
        cur_du_dt = DiscretizeForward._du_dt(cur_state, resistance, tau_m, reset, threshold)
        ctx.layer_idx = layer_idx
        ctx.save_for_backward(prv_voltage, prv_spikes, cur_state, cur_spikes, weights, cur_du_di)
        with DiscretizeForward.du_dt_dict_lock:
            DiscretizeForward.du_dt_dict.setdefault(layer_idx, cur_du_dt)
        return cur_state[..., 0], cur_state[..., 1], cur_spikes


    """
    Parameters:
    -- grad_voltage_cur: (batch_size, num_step, num_neuron_cur)
    
    Variables:
    -- prv_du_dt: (batch_size, num_step, num_neuron_prv)
    """
    @staticmethod 
    def backward(ctx, cur_grad_voltage_: torch.Tensor, cur_grad_current_: torch.Tensor, cur_grad_spikes: torch.Tensor):
        layer_idx = ctx.layer_idx
        prv_voltage, prv_spikes, cur_state, cur_spikes, weights, cur_du_di = ctx.saved_tensors      
        bias: torch.Tensor = torch.einsum("bnx,xy->bny", prv_spikes, weights)
        batch_size, num_step, num_neuron_cur = cur_grad_voltage_.shape
        num_neuron_prv = prv_spikes.shape[-1]

        step_size: float        = DiscretizeForward.step_size
        cur_tau_m: float        = getattr(DiscretizeForward, f"tau_m__{layer_idx}")
        cur_tau_s: float        = getattr(DiscretizeForward, f"tau_s__{layer_idx}")
        cur_resistance: float   = getattr(DiscretizeForward, f"resistance__{layer_idx}")
        cur_reset: float        = getattr(DiscretizeForward, f"reset__{layer_idx}")
        cur_threshold: float    = getattr(DiscretizeForward, f"threshold__{layer_idx}")
        cur_grad_voltage: torch.Tensor = torch.cat((cur_grad_voltage_, torch.zeros((batch_size, 1, num_neuron_cur), device=DiscretizeForward._device())), dim=1)                                                                                                                                                   # (batch_size, num_step + 1, num_neuron_cur)
        recur_di: torch.Tensor = DiscretizeForward._recur_di(cur_state, cur_grad_voltage_, bias, step_size, cur_resistance, cur_tau_m, cur_tau_s, cur_reset, cur_threshold)
        grad_weights: torch.Tensor = None
        if hasattr(DiscretizeForward, "discretize_grad_weight") and getattr(DiscretizeForward, "discretize_grad_weight") is True:
            grad_weights = DiscretizeForward._pe_pw_i(recur_di, weights, prv_spikes, step_size, cur_tau_s)
        else: grad_weights = DiscretizeForward._pe_pw(prv_spikes, weights, cur_grad_voltage, step_size, cur_resistance, cur_tau_m, cur_tau_s)
        DiscretizeForward._assert("grad_weights", grad_weights)
        if layer_idx == 0: return None, None, -grad_weights
        
        prv_tau_m: float        = getattr(DiscretizeForward, f"tau_m__{layer_idx - 1}")
        prv_reset: float        = getattr(DiscretizeForward, f"reset__{layer_idx - 1}")
        with DiscretizeForward.du_dt_dict_lock:
            prv_du_dt: torch.Tensor = DiscretizeForward.du_dt_dict[layer_idx - 1]
        pi_pu: torch.Tensor = DiscretizeForward._pi_pu(batch_size, num_step, num_neuron_cur, num_neuron_prv, step_size, cur_tau_s, prv_tau_m, prv_reset, 
                                                       prv_spikes, weights, prv_du_dt)
        grad_voltage_prv = DiscretizeForward._recur_du(recur_di, pi_pu[:, :-1, :-1], step_size, prv_tau_m)
        assert grad_voltage_prv.shape == (batch_size, num_step, num_neuron_prv)                                                                                                                                      # (batch_size, num_step + 1, num_neuron_prv)
        DiscretizeForward._assert("pi_pu", pi_pu)
        DiscretizeForward._assert("grad_voltage_prv", grad_voltage_prv)
        
        return grad_voltage_prv, None, -grad_weights


class SpikeCountLoss(torch.autograd.Function):
    @staticmethod
    def _du_dt(states: torch.Tensor, resistance: float, tau_m: float, reset: float, threshold: float) -> torch.Tensor:
        voltage, current = states[..., 0], states[..., 1]
        return (-voltage + resistance * current - reset * (voltage >= threshold)) / tau_m
    
    @staticmethod
    def _pu_pt(spikes: torch.Tensor, cur_step: int, step_size: float, tau_m: float, reset: float) -> torch.Tensor:
        cur_time: float = cur_step * step_size 
        batch_size, num_step, num_neuron = spikes.shape
        times_: torch.Tensor = torch.arange(num_step).view(1, num_step, 1).expand(batch_size, num_step, num_neuron).to(SpikeCountLoss.device)
        times: torch.Tensor = step_size * times_
        return (reset / (tau_m ** 2) * torch.exp(-(times - cur_time) / tau_m)) * spikes[:, cur_step].unsqueeze(1).expand(batch_size, num_step, num_neuron) * (times_ >= cur_step)
    
    @staticmethod
    def _inv_du_dt(du_dt: torch.Tensor, scale: int=8):
        inv_du_dt: torch.Tensor = 1 / du_dt
        return torch.clamp(inv_du_dt, min=-scale, max=scale)

    """
    Parameters:
    - diff_cnt_masked: (batch_size, num_step, num_neuron)
    - output_spikes:   (batch_size, num_step, num_neuron)
    
    Returns: (batch_size, num_step, num_step, num_neuron)
    """
    @staticmethod
    def _pe_pu(diff_cnt_masked: torch.Tensor, output_voltage: torch.Tensor, output_current: torch.Tensor, output_spikes: torch.Tensor, step_size: float, tau_m: float, resistance: float, threshold: float, reset: float) -> torch.Tensor:
        batch_size, num_step, num_neuron = diff_cnt_masked.shape
        pe_pu: torch.Tensor = torch.zeros((batch_size, num_step, num_step, num_neuron), device=SpikeCountLoss.device)
        pe_pu[:, torch.arange(num_step), torch.arange(num_step), :] = diff_cnt_masked
        du_dt_: torch.Tensor = SpikeCountLoss._du_dt(torch.stack((output_voltage, output_current), dim=-1), resistance, tau_m, reset, threshold)
        nxt: torch.Tensor = torch.arange(num_step).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, num_neuron).to(SpikeCountLoss.device)
        
        for cur_step in range(num_step - 1, -1, -1):                                                                                                                                                                              
            cur_du_dt: torch.Tensor = du_dt_[:, cur_step]                                                                                                                                                                
            cur_pu_pt: torch.Tensor = SpikeCountLoss._pu_pt(output_spikes, cur_step, step_size, tau_m, reset) * output_spikes[:, cur_step, :].view(batch_size, 1, num_neuron).expand(batch_size, num_step, num_neuron)                                                                                                           
            cur_pu_pt = torch.gather(cur_pu_pt, 1, nxt)                                                                                                                                                              
            coeff_1: torch.Tensor = torch.einsum("btn,bn->btn", cur_pu_pt, SpikeCountLoss._inv_du_dt(cur_du_dt))                                                                                                                                    
            coeff_2: torch.Tensor = torch.gather(pe_pu, 2, nxt.view(batch_size, num_step, 1, num_neuron)).squeeze(2)                                                         
            pe_pu[:, :, cur_step] = torch.where(nxt > cur_step, coeff_1 * coeff_2, pe_pu[:, :, cur_step])
            nxt = torch.where(output_spikes[:, cur_step].view(batch_size, 1, num_neuron).expand(batch_size, num_step, num_neuron) == 1, torch.min(nxt, torch.tensor(cur_step)), nxt)
        return pe_pu
    

    """
    Parameters:
    - pe_pu: (batch_size, num_step, num_step, num_neuron)

    Returns: (batch_size, num_step, num_neuron)    
    """
    @staticmethod 
    def _reduce_pe_pu(pe_pu: torch.Tensor, step_size: float, tau_m: float) -> torch.Tensor:
        batch_size, num_step, _, num_neuron = pe_pu.shape 
        grad_u: torch.Tensor = torch.zeros((batch_size, num_step, num_neuron)).to(SpikeCountLoss.device)
        grad_u[:, -1] = pe_pu[:, -1, -1]
        du_nxt_du = tau_m / (tau_m + step_size)
        for prv_step in range(num_step - 2, -1, -1):
            grad_u[:, prv_step] = grad_u[:, prv_step + 1] * du_nxt_du
            for nxt_step in range(prv_step, num_step):
                grad_u[:, prv_step] += pe_pu[:, nxt_step, prv_step]
        return grad_u  



    """
    Notes:
    - num_class == num_neuron
    Parameters:
    - output_spikes: (batch_size, num_step, num_class) 
    - target_spikes: (batch_size, num_class) 
    """
    @staticmethod
    def forward(ctx, output_voltage: torch.Tensor, output_current: torch.Tensor, output_spikes: torch.Tensor, target_spikes: torch.Tensor) -> float:
        batch_size, num_step, num_class = output_spikes.shape
        output_spikes_cnt: torch.Tensor = torch.sum(output_spikes, dim=1)
        ctx.save_for_backward(output_voltage, output_current, output_spikes, target_spikes)
        return torch.sum((output_spikes_cnt - target_spikes) ** 2, dim=(0, 1)) / batch_size
    

    """
    Returns: (batch_size, num_step + 1, num_class)

    Variables:
    - diff_cnt: (batch_size, num_class)
    """
    @staticmethod
    def backward(ctx, grad_outputs) -> torch.Tensor:
        output_voltage, output_current, output_spikes, target_spikes = ctx.saved_tensors
        batch_size, num_step, num_class = output_spikes.shape
        output_spikes_cnt: torch.Tensor = torch.sum(output_spikes, dim=1)
        diff_cnt_: torch.Tensor = (target_spikes - output_spikes_cnt)  / num_step
        diff_cnt_masked: torch.Tensor = diff_cnt_.unsqueeze(1).repeat(1, num_step, 1)
        pe_pu: torch.Tensor = SpikeCountLoss._pe_pu(diff_cnt_masked, output_voltage, output_current, output_spikes, SpikeCountLoss.step_size, SpikeCountLoss.tau_m, SpikeCountLoss.resistance, SpikeCountLoss.threshold, SpikeCountLoss.reset)
        return SpikeCountLoss._reduce_pe_pu(pe_pu, SpikeCountLoss.step_size, SpikeCountLoss.tau_m), None, None, None 
    

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
        




        

        



