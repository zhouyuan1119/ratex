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
from ratex.utils.mem_model_utils import analyze_training_peak_memory, profile_training_peak_memory, random_torch_tensor

class TorchLeNet(nn.Module):
    def __init__(self, input_shape=28, num_classes=10):
        super(TorchLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, bias=False)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, bias=False)
        self.linear1 = nn.Linear(((input_shape // 2 - 4) // 2) ** 2 * 16, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = torch.relu(out)
        out = F.avg_pool2d(out, (2, 2), (2, 2))
        out = self.conv2(out)
        out = torch.relu(out)  # pylint: disable=no-member
        out = F.avg_pool2d(out, (2, 2), (2, 2))
        out = torch.flatten(out, 1)  # pylint: disable=no-member
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        return out

def main():
    model_lt = TorchLeNet()
    model_cuda = copy.deepcopy(model_lt)
    model_lt = model_lt.to(device="lazy", dtype=torch.float32)
    optimizer_lt = torch.optim.SGD(model_lt.parameters(), lr=0.001)
    loss_fn = torch.nn.NLLLoss()
    peak_memory_mbs = analyze_training_peak_memory(
        model_lt, optimizer_lt, loss_fn, (4, 1, 28, 28), (4,), torch.float32, torch.int64, output_range=[0, 10])
    model_cuda = model_cuda.cuda()
    optimizer = optim.SGD(model_cuda.parameters(), lr=0.001)
    peak_memory_bs = profile_training_peak_memory(
        model_cuda, optimizer, torch.nn.NLLLoss(), (4, 1, 28, 28), (4,), torch.float32, torch.int64, output_range=[0, 10])

if __name__ == "__main__":
    main()
