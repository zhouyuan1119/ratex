# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import torch
import ratex
import ratex.lazy_tensor_core.debug.metrics as metrics
import ratex.lazy_tensor_core.core.lazy_model as lm
import _RATEXC
_RATEXC._set_ratex_vlog_level(-5)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from ratex.utils.mem_model_utils import analyze_training_peak_memory, profile_training_peak_memory
import transformers
class TorchMLP(nn.Module):
    def __init__(self, input_shape=256, num_classes=8):
        super(TorchMLP, self).__init__()
        self.linear1 = nn.Linear(input_shape, 4096, bias=False)
        self.linear2 = nn.Linear(4096, 8192, bias=False)
        self.linear3 = nn.Linear(8192, 2048, bias=False)
        self.linear4 = nn.Linear(2048, 256, bias=False)
        self.linear5 = nn.Linear(256, num_classes, bias=False)

    def forward(self, x):
        out = self.linear1(x)
        out = torch.relu(out)  # pylint: disable=no-member
        out = self.linear2(out)
        out = torch.relu(out)
        out = self.linear3(out)
        out = torch.relu(out)
        out = self.linear4(out)
        out = torch.relu(out)
        out = self.linear5(out)
        return out

def main():
    model_lt = TorchMLP()
    model_cuda = copy.deepcopy(model_lt)
    model_lt = model_lt.to(device="lazy", dtype=torch.float32)
    optimizer_lt = torch.optim.SGD(model_lt.parameters(), lr=0.001)
    loss_fn = torch.nn.NLLLoss()
    peak_memory_mbs = analyze_training_peak_memory(
        model_lt, optimizer_lt, loss_fn, (64, 256), (64,), torch.float32, torch.int64, [0, 8])
    model_cuda = model_cuda.cuda()
    optimizer = optim.SGD(model_cuda.parameters(), lr=0.001)
    peak_memory_bs = profile_training_peak_memory(
        model_cuda, optimizer, torch.nn.NLLLoss(), (64, 256), (64,), torch.float32, torch.int64, [0, 8])

if __name__ == "__main__":
    main()
