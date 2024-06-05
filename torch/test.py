from itertools import product
from function import DiscretizeForward
import numpy as np 
import torch 
from typing import Callable, Dict, Tuple


class VectorTest:
    def __init__(
        self, batch_size: int, step_size: float, num_step: int, num_neuron_prv: int, num_neuron_cur: int,
        resistance: float, tau_s: float, tau_m: float, reset: float, threshold: float,
        num_iter: int = 10
    ):
        for name, var in vars().items(): 
            if name != "self":
                setattr(self, name, var)
        self.state_shape_cur: Tuple = (batch_size, num_step, num_neuron_cur)
        self.state_shape_prv: Tuple = (batch_size, num_step, num_neuron_prv)
    
    def vec_test(self, func: Callable):
        func_name: str = func.__name__

        if func_name == "_du_di":
            for it in range(self.num_iter):
                states: torch.Tensor = torch.rand((*self.state_shape_cur, 2))
                bias: torch.Tensor = torch.rand(self.state_shape_cur)
                res_: torch.Tensor = func(states, bias, self.resistance, self.tau_s, self.tau_m, self.reset, self.threshold)
                assert res_.shape == self.state_shape_cur
                ans_: torch.Tensor = torch.zeros(self.state_shape_cur)
                for b, t, n_c in product(range(self.batch_size), range(self.num_step), range(self.num_neuron_cur)):
                    ans_[b][t][n_c] = self.tau_s / self.tau_m * (-states[b][t][n_c][0] + self.resistance * states[b][t][n_c][1] - self.reset * int(states[b][t][n_c][0] >= self.threshold)) / (-states[b][t][n_c][1] + bias[b][t][n_c])
                assert torch.allclose(res_, ans_)
        
        elif func_name == "_du_dt":
            for it in range(self.num_iter):
                states: torch.Tensor = torch.rand((*self.state_shape_cur, 2))
                res_: torch.Tensor = func(states, self.resistance, self.tau_m, self.reset, self.threshold)
                assert res_.shape == self.state_shape_cur
                ans_: torch.Tensor = torch.zeros(self.state_shape_cur)
                for b, t, n_c in product(range(self.batch_size), range(self.num_step), range(self.num_neuron_cur)):
                    ans_[b][t][n_c] = 1 / self.tau_m * (-states[b][t][n_c][0] + self.resistance * states[b][t][n_c][1] - self.reset * int(states[b][t][n_c][0] >= self.threshold))
                assert torch.allclose(res_, ans_)

        elif func_name == "_pu_pt":
            for it in range(self.num_iter):
                spikes: torch.Tensor = torch.randint(0, 2, self.state_shape_cur)
                cur_step: int = torch.randint(0, self.num_step, (1,)).item()
                res_: torch.Tensor = func(spikes, cur_step, self.step_size, self.tau_m, self.reset)
                assert res_.shape == self.state_shape_cur
                ans_: torch.Tensor = torch.zeros(self.state_shape_cur)
                for b, t, n_c in product(range(self.batch_size), range(self.num_step), range(self.num_neuron_cur)):
                    if t < cur_step or spikes[b, cur_step, n_c] == 0: continue
                    ans_[b][t][n_c] = self.reset / (self.tau_m ** 2) * np.exp(-(t * self.step_size - cur_step * self.step_size) / self.tau_m)
                assert torch.allclose(res_, ans_)
        
        elif func_name == "_pi_pt":
            for it in range(self.num_iter):
                prv_spikes = torch.randint(0, 2, self.state_shape_prv)
                weights = torch.rand((self.num_neuron_prv, self.num_neuron_cur))
                cur_step: int = torch.randint(0, self.num_step, (1,)).item()
                res_: torch.Tensor = func(prv_spikes, weights, cur_step, self.step_size, self.tau_s)
                assert res_.shape == self.state_shape_cur
                ans_: torch.Tensor = torch.zeros(self.state_shape_cur)
                for b, t, n_p, n_c in product(range(self.batch_size), range(self.num_step), range(self.num_neuron_prv), range(self.num_neuron_cur)):
                    if t < cur_step or prv_spikes[b][cur_step][n_p] == 0: continue 
                    ans_[b][t][n_c] += weights[n_p][n_c] / (self.tau_s ** 2) * np.exp(- (t * self.step_size - cur_step * self.step_size) / self.tau_s)
                assert torch.allclose(res_, ans_)

        elif func_name == "_pu_pw":
            for it in range(self.num_iter):
                prv_spikes = torch.randint(0, 2, self.state_shape_prv)
                weights = torch.rand((self.num_neuron_prv, self.num_neuron_cur))
                res_ = func(prv_spikes, weights, self.step_size, self.resistance, self.tau_m, self.tau_s)
                shape_: Tuple = (self.batch_size, self.num_step, self.num_neuron_prv, self.num_neuron_cur)
                assert res_.shape == shape_
                ans_ = torch.zeros(shape_)
                for b, t_p, t_c, n_p, n_c in product(range(self.batch_size), range(self.num_step), range(self.num_step), range(self.num_neuron_prv), range(self.num_neuron_cur)):
                    if t_p > t_c or prv_spikes[b][t_p][n_p] == 0: continue
                    t_p_, t_c_ = t_p * self.step_size, t_c * self.step_size
                    ans_[b][t_c][n_p][n_c] += self.resistance / (self.tau_m - self.tau_s) * (np.exp(- (t_c_ - t_p_) / self.tau_m) - np.exp(- (t_c_ - t_p_) / self.tau_s))
                assert torch.allclose(res_, ans_)

        elif func_name == "_pe_pw":
            for it in range(self.num_iter):
                prv_spikes = torch.randint(0, 2, self.state_shape_prv)
                w_shape = (self.num_neuron_prv, self.num_neuron_cur)
                weights = torch.rand(w_shape)
                grad_u = torch.rand((self.batch_size, self.num_step + 1, self.num_neuron_cur))
                grad_u[:, self.num_step] = 0
                res_ = func(prv_spikes, weights, grad_u, self.step_size, self.resistance, self.tau_m, self.tau_s)
                assert res_.shape == w_shape
                ans_ = torch.zeros(w_shape)
                for b, t_p, t_c, n_p, n_c in product(range(self.batch_size), range(self.num_step), range(self.num_step), range(self.num_neuron_prv), range(self.num_neuron_cur)):
                    if t_p > t_c or prv_spikes[b][t_p][n_p] == 0: continue
                    t_p_, t_c_ = t_p * self.step_size, t_c * self.step_size
                    tmp_val =  self.resistance / (self.tau_m - self.tau_s) * (np.exp(- (t_c_ - t_p_) / self.tau_m) - np.exp(- (t_c_ - t_p_) / self.tau_s))
                    ans_[n_p][n_c] += tmp_val * grad_u[b][t_c][n_c]
                assert torch.allclose(res_, ans_)

        elif func_name == "_pi_pu":
            for it in range(self.num_iter):
                prv_spikes = torch.randint(0, 2, self.state_shape_prv)
                w_shape = (self.num_neuron_prv, self.num_neuron_cur)
                weights = torch.rand(w_shape)
                prv_du_dt = torch.rand(self.state_shape_prv) * 10
                cur_tau_s, prv_tau_m, prv_reset = self.tau_s, self.tau_m, self.reset
                shape_: Tuple = (self.batch_size, self.num_step + 1, self.num_step + 1, self.num_neuron_cur, self.num_neuron_prv)
                
                res_ = func(self.batch_size, self.num_step, self.num_neuron_cur, self.num_neuron_prv, self.step_size, cur_tau_s, prv_tau_m, prv_reset,
                            prv_spikes, weights, prv_du_dt)
                assert res_.shape == shape_

                ans_: torch.Tensor = torch.zeros(shape_)
                last_spikes = torch.arange(self.num_step + 1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(self.batch_size, 1, self.num_neuron_cur, self.num_neuron_prv)
                for b, t_p, t_c, n_p, n_c in product(range(self.batch_size), range(self.num_step - 1, -1, -1), range(self.num_step), range(self.num_neuron_prv), range(self.num_neuron_cur)):
                    if t_c < t_p or prv_spikes[b][t_p][n_p] == 0: continue 
                    t_p_, t_c_ = t_p * self.step_size, t_c * self.step_size
                    pi_pt = weights[n_p][n_c] / (cur_tau_s ** 2) * np.exp(- (t_c_ - t_p_) / cur_tau_s) 
                    du_dt = prv_du_dt[b, t_p, n_p]
                    pi_pu = pi_pt / du_dt
                    ans_[b][t_c][t_p][n_c][n_p] = pi_pu
                    if t_c != t_p:
                        last_spike = last_spikes[b][t_c][n_c][n_p]
                        pu_pt = prv_reset / (prv_tau_m ** 2) * np.exp(- (last_spike * self.step_size - t_p_) / prv_tau_m)
                        ans_[b][t_c][t_p][n_c][n_p] += ans_[b][t_c][last_spike][n_c][n_p] * pu_pt / du_dt
                    last_spikes[b][t_c][n_c][n_p] = np.minimum(last_spikes[b][t_c][n_c][n_p], t_p)

                assert torch.allclose(res_, ans_)

