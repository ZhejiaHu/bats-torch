from experiments.mnist.Dataset import Dataset as MNIST_Dataset
from function import DiscretizeForward, SpikeCountLoss
from network import MNIST_Network
import numpy as np
from itertools import product
import random 
from test import learning_test
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Dict, Tuple
  

def mnist_train(num_layer: int, num_step: int, step_size: float, forward_type: int, hyper_params: Dict, device: str="cuda", pre_test: bool=False, pre_train_num: int=3, pre_test_num: int=3):

    MNIST_DATASET_PTH: str = "../datasets/mnist.npz"
    MNIST__NUM_CLASSES: int = 10
    MNIST__LEARNING_RATE: float = 0.01
    MNIST__TRAIN_EPOCHS: int = 50
    MNIST__NUM_TRAIN_SAMPLES = 60000
    MNIST__NUM_TEST_SAMPLES = 10000
    MNIST__TRAIN_BATCH_SIZE: int = 5
    MNIST__NUM_TRAIN_BATCH = int(MNIST__NUM_TRAIN_SAMPLES / MNIST__TRAIN_BATCH_SIZE)
    MNIST__TEST_BATCH_SIZE: int = 10
    MNIST__NUM_TEST_BATCH = int(MNIST__NUM_TEST_SAMPLES / MNIST__TEST_BATCH_SIZE) 
    MNIST__TEST_PERIOD = 3 
    MNIST__TARGET_TRUE = 15
    MNIST__TARGET_FALSE = 3

    """
    Returns:
    - ipt_spikes: (batch_size, num_step, num_neuron)
    - labels: (batch_size, )
    - tgt_spikes_: (batch_size, num_class)
    """
    def mnist_get_batch(train: bool, dataset: MNIST_Dataset, batch_idx: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spikes_, _, labels_ = dataset.get_train_batch(batch_idx, batch_size) if train else dataset.get_test_batch(batch_idx, batch_size)
        batch_size, num_neuron = spikes_.shape[:-1]
        spikes_ = (spikes_.squeeze(-1) / np.max(spikes_) * (num_step - 1)).astype(int)
        spikes_ = num_step - 1 - spikes_
        spikes = np.zeros((batch_size, num_step, num_neuron))
        for b, n in product(range(batch_size), range(num_neuron)):
            spikes[b, :spikes_[b, n], n] = 1
        tgt_spikes_: np.ndarray = np.full((batch_size, MNIST__NUM_CLASSES), MNIST__TARGET_FALSE)
        tgt_spikes_[np.arange(batch_size), labels_] = MNIST__TARGET_TRUE
        return torch.from_numpy(spikes).to(device), torch.from_numpy(labels_).to(device), torch.from_numpy(tgt_spikes_).to(device) 

    dataset = MNIST_Dataset(MNIST_DATASET_PTH)
    network = MNIST_Network(num_layer, num_step, step_size, forward_type, hyper_params, device)
    optimizer = Adam(network.parameters(), lr=MNIST__LEARNING_RATE)
    
    if pre_test:
        train_idxs = np.random.randint(0, MNIST__TRAIN_BATCH_SIZE, pre_train_num)
        test_idxs = np.random.randint(0, MNIST__TEST_BATCH_SIZE, pre_test_num)
        train_batches = list(map(lambda idx: mnist_get_batch(True, dataset, idx, MNIST__TRAIN_BATCH_SIZE), train_idxs))
        test_batches = list(map(lambda idx: mnist_get_batch(False, dataset, idx, MNIST__TEST_BATCH_SIZE), test_idxs))
        print(f"[Pre-test] Train indices : {train_idxs} | Test indices : {test_idxs}")
        learning_test(network, MNIST__TRAIN_BATCH_SIZE, MNIST__TEST_BATCH_SIZE, train_batches, test_batches)
    
    network = MNIST_Network(num_layer, num_step, step_size, forward_type, hyper_params, device)
    optimizer = Adam(network.parameters(), lr=MNIST__LEARNING_RATE)
    for epoch in range(MNIST__TRAIN_EPOCHS):
        network.train()
        dataset.shuffle()
        num_correct = 0
        losses = 0

        for batch_idx in range(MNIST__NUM_TRAIN_BATCH):
            optimizer.zero_grad()
            ipt_spikes, labels, tgt_spikes = mnist_get_batch(True, dataset, batch_idx, MNIST__TRAIN_BATCH_SIZE)
            opt_voltage, opt_current, opt_spikes = network(ipt_spikes)
            loss = SpikeCountLoss.apply(opt_voltage, opt_current, opt_spikes, tgt_spikes)
            num_correct += SpikeCountLoss.accuracy(opt_spikes, labels) 
            loss.backward()
            optimizer.step()
            losses += loss.item()

            if batch_idx > 0 and batch_idx % MNIST__TEST_PERIOD == 0:
                print(f"[TRAIN] Epoch {epoch} --- has losses {losses} and number of correct predictions {num_correct} and accuracy rate : {num_correct / ((batch_idx + 1) * MNIST__TRAIN_BATCH_SIZE)}")
                network.eval()
                num_correct_ = 0
                losses_ = 0
                with torch.no_grad():
                    for batch_idx in range(MNIST__NUM_TEST_BATCH):
                        ipt_spikes, labels, tgt_spikes = mnist_get_batch(False, dataset, batch_idx, MNIST__TEST_BATCH_SIZE)
                        opt_voltage, opt_current, opt_spikes = network(ipt_spikes)
                        losses_ += SpikeCountLoss.apply(opt_voltage, opt_current, opt_spikes, tgt_spikes).item()
                        num_correct_ += SpikeCountLoss.accuracy(opt_spikes, labels)
                print(f"    [TEST] Epoch {epoch} --- has losses {losses_} and number of correct predictions {num_correct_} and accuracy rate : {num_correct_ / MNIST__NUM_TEST_SAMPLES}")
    










