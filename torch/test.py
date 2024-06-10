from itertools import product
from function import DiscretizeForward, SpikeCountLoss
import numpy as np 
import torch 
import torch.nn as nn
from torch.optim import Adam 
from typing import Callable, Dict, List, Tuple


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
                    if t_c != t_p and last_spikes[b][t_c][n_c][n_p] != t_p:
                        last_spike = last_spikes[b][t_c][n_c][n_p]
                        pu_pt = prv_reset / (prv_tau_m ** 2) * np.exp(- (last_spike * self.step_size - t_p_) / prv_tau_m)
                        ans_[b][t_c][t_p][n_c][n_p] += ans_[b][t_c][last_spike][n_c][n_p] * pu_pt / du_dt
                    last_spikes[b][t_c][n_c][n_p] = np.minimum(last_spikes[b][t_c][n_c][n_p], t_p)

                assert torch.allclose(res_, ans_)
        elif func_name == "_recur_di":
            for it in range(self.num_iter):
                print(it)
                states_cur: torch.Tensor = torch.rand((*self.state_shape_cur, 2))
                grad_voltage_cur: torch.Tensor = torch.rand(self.state_shape_cur)
                bias: torch.Tensor = torch.rand(self.state_shape_cur)
                res_: torch.Tensor = func(states_cur, grad_voltage_cur, bias, self.step_size, self.resistance, self.tau_m, self.tau_s, self.reset, self.threshold)
                assert res_.shape == self.state_shape_cur
                
                ans_ = torch.zeros(self.state_shape_cur)
                du_di = self.resistance * self.step_size / (self.step_size + self.tau_m) 
                di_nxt_di = self.tau_s / (self.step_size + self.tau_s)
                for b, n in product(range(self.batch_size), range(self.num_neuron_cur)):
                    ans_[b, -1, n] = grad_voltage_cur[b, -1, n] * du_di
                for t, b, n in product(range(self.num_step - 2, -1, -1), range(self.batch_size), range(self.num_neuron_cur)):
                    ans_[b, t, n] = grad_voltage_cur[b, t, n] * du_di + ans_[b, t + 1, n] * di_nxt_di
                assert torch.allclose(ans_, res_)
        elif func_name == "_recur_du":
            for it in range(self.num_iter):
                print(it)
                recur_di: torch.Tensor = torch.rand(self.state_shape_cur)
                pi_pu: torch.Tensor = torch.rand((self.batch_size, self.num_step, self.num_step, self.num_neuron_cur, self.num_neuron_prv))
                res_: torch.Tensor = func(recur_di, pi_pu, self.step_size, self.tau_m)
                assert res_.shape == self.state_shape_prv

                ans_: torch.Tensor = torch.zeros(self.state_shape_prv)
                du_nxt_du = self.tau_m / (self.step_size + self.tau_m)
                for b, n_cur, n_prv in product(range(self.batch_size), range(self.num_neuron_cur), range(self.num_neuron_prv)):
                    ans_[b][-1][n_prv] += recur_di[b][-1][n_cur] * pi_pu[b][-1][-1][n_cur][n_prv]
                for t_prv in range(self.num_step - 2, -1, -1):
                    for b, n_prv in product(range(self.batch_size), range(self.num_neuron_prv)):
                        ans_[b][t_prv][n_prv] = ans_[b][t_prv + 1][n_prv] * du_nxt_du
                    for t_cur in range(t_prv, self.num_step):
                        for b, n_cur, n_prv in product(range(self.batch_size), range(self.num_neuron_cur), range(self.num_neuron_prv)):
                            ans_[b][t_prv][n_prv] += pi_pu[b][t_cur][t_prv][n_cur][n_prv] * recur_di[b][t_cur][n_cur]
                assert torch.allclose(ans_, res_)
        elif func_name == "_pe_pu":
            for it in range(self.num_iter):
                diff_cnt_masked: torch.Tensor = torch.rand(self.state_shape_cur)
                output_voltage: torch.Tensor = torch.rand(self.state_shape_cur)
                output_current: torch.Tensor = torch.rand(self.state_shape_cur)
                output_spikes: torch.Tensor = torch.randint(0, 2, self.state_shape_cur)
                res_ = func(diff_cnt_masked, output_voltage, output_current, output_spikes, self.step_size, self.tau_m, self.resistance, self.threshold, self.reset) 
                shape_: Tuple = (self.batch_size, self.num_step, self.num_step, self.num_neuron_cur)
                assert res_.shape == shape_

                ans_: torch.Tensor = torch.zeros(shape_)
                nxt = torch.zeros(self.state_shape_cur, dtype=torch.int32)
                for i in range(self.num_step): 
                    nxt[:, i, :] = i
                    ans_[:, i, i, :] = diff_cnt_masked[:, i, :]
                for cur_step, b, step, n in product(range(self.num_step - 1, -1, -1), range(self.batch_size), range(self.num_step), range(self.num_neuron_cur)):
                    if step > cur_step and output_spikes[b, cur_step, n] == 1:
                        pu_pt_ = self.reset / (self.tau_m ** 2) * np.exp(- (nxt[b, step, n] * self.step_size - cur_step * self.step_size) / self.tau_m)
                        du_dt_ = (-output_voltage[b, cur_step, n] + self.resistance * output_current[b, cur_step, n] + self.reset * (output_voltage[b, cur_step, n] >= self.threshold)) / self.tau_m 
                        ans_[b, step, cur_step, n] = ans_[b, step, nxt[b, step, n], n] * pu_pt_ * torch.clamp(1 / du_dt_, min=-8, max=8)
                    if output_spikes[b, cur_step, n] == 1: nxt[b, step, n] = torch.min(nxt[b, step, n], torch.tensor(cur_step))
                assert torch.allclose(ans_, res_) 

    

"""
Repetitive testing on one dataset. 
"""
Batches = List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
def learning_test(model: nn.Module, train_batch_size: int, test_batch_size: int, train_batches: Batches, test_batches: Batches, num_epoch: int=100): 
    optimizer = Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epoch):
        model.train()
        losses_train = 0
        accuracies_train = 0
        
        for batch in train_batches:
            optimizer.zero_grad()
            ipt_spikes, labels, tgt_spikes = batch
            opt_voltage, opt_current, opt_spikes = model(ipt_spikes)
            loss = SpikeCountLoss.apply(opt_voltage, opt_current, opt_spikes, tgt_spikes)
            accuracy = SpikeCountLoss.accuracy(opt_spikes, labels)      
            losses_train += loss.item()
            accuracies_train += accuracy.item()
            loss.backward()
            optimizer.step()
        avg_loss_train = losses_train / len(train_batches)
        avg_accuracy_train = accuracies_train / (train_batch_size * len(train_batches))
        
        print(f"[Train] Epoch {epoch} has average loss {avg_loss_train}, total correct {accuracies_train}, and percentage accuracy {avg_accuracy_train * 100:.2f}%")
        
        model.eval()
        losses_test = 0
        accuracies_test = 0
        
        with torch.no_grad():
            for batch in test_batches:
                ipt_spikes, labels, tgt_spikes = batch
                opt_voltage, opt_current, opt_spikes = model(ipt_spikes)
                loss = SpikeCountLoss.apply(opt_voltage, opt_current, opt_spikes, tgt_spikes)
                accuracy = SpikeCountLoss.accuracy(opt_spikes, labels)
                losses_test += loss.item()
                accuracies_test += accuracy.item()
        avg_loss_test = losses_test / len(test_batches)
        avg_accuracy_test = accuracies_test / (test_batch_size * len(test_batches))
        print(f"[Test] Epoch {epoch} has average loss {avg_loss_test}, total correct {accuracies_test}, and percentage accuracy {avg_accuracy_test * 100:.2f}%")
        
        model.train()

        
                
    print("Learning test passes")
