from copy import deepcopy
import datetime
from function import DiscretizeForward, SpikeCountLoss
from itertools import product
from network import MNIST_Network
import numpy as np
import random 
from snntorch import spikegen
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Callable, Dict, List, Tuple


def save_checkpoint(state: Dict, epoch: int, is_best: bool, filename_: str) -> None:
    torch.save(state, filename_ + f"_lastest_{epoch}.pth.tar")
    if is_best:
        torch.save(state, filename_ + f"_best_{epoch}.pth.tar")


def load_checkpoint(filename: str) -> Tuple[nn.Module, Optimizer, LRScheduler, Dict]:
    checkpoint = torch.load(filename)
    network = MNIST_Network(checkpoint["num_layer"], checkpoint["num_step"], checkpoint["step_size"], checkpoint["forward_type"], checkpoint["hyper_params"])
    network.load_state_dict(checkpoint["state_dict"])
    optimizer = Adam(network.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    scheduler.load_state_dict(checkpoint["scheduler"])
    return network, optimizer, scheduler, checkpoint


def mnist_train(num_layer: int, num_step: int, step_size: float, forward_type: int, hyper_params: Dict, device: str="cuda", encoder="amplitude", load_filename=None):
    DATASET_NAME = "MNIST"

    MNIST__DATASET_TRAIN_PTH: str = "datasets/mnist/train"
    MNIST__DATASET_TEST_PTH: str = "datasets/mnist/test"
    MNIST__NUM_CLASSES: int = 10
    MNIST__TRAIN_EPOCHS: int = 50
    MNIST__NUM_TRAIN_SAMPLES = 60000
    MNIST__NUM_TEST_SAMPLES = 10000
    MNIST__TRAIN_BATCH_SIZE: int = 50
    MNIST__NUM_TRAIN_BATCH = int(MNIST__NUM_TRAIN_SAMPLES / MNIST__TRAIN_BATCH_SIZE)
    MNIST__TEST_BATCH_SIZE: int = 100
    MNIST__TEST_PERIOD = int(MNIST__NUM_TRAIN_BATCH * 0.1)
    MNIST__TARGET_TRUE = num_step - 5
    MNIST__TARGET_FALSE = 0

    """
    Convert from snntorch spike shape to our spike shape
    - Input (num_step, batch_size, 1, 28, 28)
    - Output (batch_step, num_step, 28 * 28)
    """
    _convert: Callable = lambda raw_spike, batch_size: raw_spike.squeeze(2).transpose(0, 1).reshape(batch_size, num_step, -1)
    
    """
    Parameters:
    - labels: (batch_size,)
    """
    def _gen_ipt_spikes(data: torch.Tensor) -> torch.Tensor:
        batch_size, num_neuron = data.shape[0], data.shape[-2] * data.shape[-1] 
        if encoder == "amplitude":
            data_ = (data.squeeze(1).reshape(batch_size, -1) * (num_step - 1)).to(torch.int32)
            spikes = torch.zeros((batch_size, num_step, num_neuron))
            mid = torch.arange(num_step).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, num_neuron)
            spikes = torch.where(mid <= data_.view(batch_size, 1, num_neuron).expand(batch_size, num_step, num_neuron), 1, 0)
            #spikes_ = torch.zeros((batch_size, num_step, num_neuron))
            #for b, n in product(range(batch_size), range(num_neuron)):
            #    spikes_[b, :data_[b, n]+1, n] = 1
            #assert torch.allclose(spikes.to(torch.float32), spikes_.to(torch.float32))
            return spikes
        elif encoder == "rate": return _convert(spikegen.rate(data, num_steps=num_step), batch_size)
        else: return _convert(spikegen.latency(data, num_steps=num_step, normalize=True), batch_size)
                                                


    def _gen_label_spikes(batch_size: int, labels: torch.Tensor) -> torch.Tensor:
        tgt_spikes: np.ndarray = np.full((batch_size, MNIST__NUM_CLASSES), MNIST__TARGET_FALSE)
        tgt_spikes[np.arange(batch_size), labels] = MNIST__TARGET_TRUE
        return torch.from_numpy(tgt_spikes)


    def _load_data(transform: transforms.Compose=transforms.Compose([
        transforms.Resize((28,28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))])
    ):
        mnist_train = datasets.MNIST(MNIST__DATASET_TRAIN_PTH, train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(MNIST__DATASET_TEST_PTH, train=False, download=True, transform=transform)
        return DataLoader(mnist_train, batch_size=MNIST__TRAIN_BATCH_SIZE, shuffle=True), DataLoader(mnist_test, batch_size=MNIST__TEST_BATCH_SIZE)

    
    def _store_state(model: nn.Module, optimizer: Optimizer, scheduler: LRScheduler, epoch: int, log: List[Tuple], is_best: bool=False) -> Dict:
        state: Dict = {
            "dataset": DATASET_NAME, "epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "num_layer": num_layer, "num_step": num_step, "step_size": step_size, "forward_type": forward_type, "hyper_params": hyper_params, "encoder" : encoder, "target_true": MNIST__TARGET_TRUE, "target_false": MNIST__TARGET_FALSE,
            "log": log
        }
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_: str = f"{DATASET_NAME}__{now}__EPOCH_{epoch}"
        save_checkpoint(state, epoch, is_best, filename_)   


    def _phase_plane(labels: torch.Tensor, opt_voltage: torch.Tensor, opt_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_step, num_class = opt_voltage.shape
        opt_v_c_glob, opt_i_c_glob, opt_v_w_glob, opt_i_w_glob = torch.zeros(num_step), torch.zeros(num_step), torch.zeros(num_step), torch.zeros(num_step)
        for batch_idx in range(batch_size):
            label  = labels[batch_idx].item()
            opt_v, opt_i = opt_voltage[batch_idx], opt_current[batch_idx]
            opt_v_c, opt_i_c = opt_v[:, label], opt_i[:, label]
            opt_v_w, opt_i_w = opt_v[:, [i for i in range(num_class) if i != label]].mean(dim=1), opt_i[:, [i for i in range(num_class) if i != label]].mean(dim=1)
            opt_v_c_glob += opt_v_c; opt_i_c_glob += opt_i_c; opt_v_w_glob += opt_v_w; opt_i_w_glob += opt_i_w
        return opt_v_c_glob / batch_size, opt_i_c_glob / batch_size, opt_v_w_glob / batch_size, opt_i_w_glob / batch_size
        

    train_loader, test_loader = _load_data()
    if load_filename is None:
        network = MNIST_Network(num_layer, num_step, step_size, forward_type, hyper_params, device) 
        optimizer = Adam(network.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    else: 
        network, optimizer, scheduler, checkpoint = load_checkpoint(load_filename)
        MNIST__TARGET_TRUE, MNIST__TARGET_FALSE = checkpoint["target_true"], checkpoint["target_false"]
    
    # Tuple[Training Loss, Training Accuracy, Testing Loss, Testing Accuracy] 
    log: List[Tuple[float, float, float, float]] = []


    for epoch in range(MNIST__TRAIN_EPOCHS):
        network.train()
        num_correct = 0
        losses = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            tgt_spikes = _gen_label_spikes(MNIST__TRAIN_BATCH_SIZE, labels).to(device).to(torch.float32)
            ipt_spikes = _gen_ipt_spikes(data).to(device).to(torch.float32)
            labels = labels.to(device).to(torch.uint8)
            opt_voltage, opt_current, opt_spikes = network(ipt_spikes)
            loss = SpikeCountLoss.apply(opt_voltage, opt_current, opt_spikes, tgt_spikes)
            num_correct += SpikeCountLoss.accuracy(opt_spikes, labels) 
            loss.backward()
            optimizer.step()
            losses += loss.item()

            if (batch_idx > 0 or epoch > 0) and batch_idx % MNIST__TEST_PERIOD == 0:
                if batch_idx == 0: scheduler.step(loss)
                print(f"[TRAIN] Epoch {epoch} --- has losses {losses} and number of correct predictions {num_correct} and accuracy rate : {num_correct / (MNIST__TEST_PERIOD * MNIST__TRAIN_BATCH_SIZE)}")
                opt_v_c, opt_i_c, opt_v_w, opt_i_w = _phase_plane(labels.detach().cpu(), opt_voltage.detach().cpu(), opt_current.detach().cpu())
                
                network.eval()
                cur_log = [deepcopy(losses), deepcopy(num_correct)]
                num_correct = 0
                losses = 0
                with torch.no_grad():
                    num_correct_ = 0
                    losses_ = 0
                    for batch_idx_, (data_, labels_) in enumerate(test_loader):
                        tgt_spikes_ = _gen_label_spikes(MNIST__TEST_BATCH_SIZE, labels_).to(device).to(torch.float32)
                        ipt_spikes_ = _gen_ipt_spikes(data_).to(device).to(torch.float32)
                        labels_ = labels_.to(device).to(torch.uint8)
                        opt_voltage_, opt_current_, opt_spikes_ = network(ipt_spikes_)
                        losses_ += SpikeCountLoss.apply(opt_voltage_, opt_current_, opt_spikes_, tgt_spikes_).item()
                        num_correct_ += SpikeCountLoss.accuracy(opt_spikes_, labels_)
                cur_log.append(deepcopy(losses_))
                cur_log.append(deepcopy(num_correct_))
                log.append(tuple(cur_log))
                print(f"    [TEST] Epoch {epoch} --- has losses {losses_} and number of correct predictions {num_correct_} and accuracy rate : {num_correct_ / MNIST__NUM_TEST_SAMPLES}")
        
        _store_state(network, optimizer, scheduler, epoch, log, is_best=False)
            
        









