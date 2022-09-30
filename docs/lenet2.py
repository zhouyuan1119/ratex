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
from ratex.utils.mem_model_utils import analyze_training_peak_memory, print_mem_breakdown, profile_training_peak_memory, wrap_model
import accelerate

class TorchLeNet(nn.Module):
    def __init__(self, input_shape=28, num_classes=10):
        super(TorchLeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, bias=False)
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
        out = torch.relu(out)  # pylint: disable=no-member
        out = self.linear2(out)
        out = torch.relu(out)  # pylint: disable=no-member
        out = self.linear3(out)
        return out

def main():
    with accelerate.init_empty_weights():
        model_lt = TorchLeNet()
    # model_lt = TorchLeNet()
    model_lt.train()
    model_lt = wrap_model(model_lt, 'top')
    model_lt = model_lt.to(device="lazy", dtype=torch.float32)
    optimizer_lt = torch.optim.SGD(model_lt.parameters(), lr=0.001)
    loss_fn = torch.nn.NLLLoss()
    peak_memory_ltc, mem_breakdown, ir_info = analyze_training_peak_memory(
        model_lt, optimizer_lt, loss_fn, (4, 3, 32, 32), (4,), torch.float32, torch.int64, output_range=[0, 10],
        n_batches=1)
    
    model_cuda = TorchLeNet()
    model_cuda.train()
    model_cuda = model_cuda.cuda()
    optimizer = optim.SGD(model_cuda.parameters(), lr=0.001)
    peak_memory_profiled = profile_training_peak_memory(
        model_cuda, optimizer, torch.nn.NLLLoss(), (4, 3, 32, 32), (4,), torch.float32, torch.int64, output_range=[0, 10])
    print('Profiled peak memory: {0:6.2f} MBs'.format(peak_memory_profiled))
    print('Analyzed peak memory: {0:6.2f} MBs'.format(peak_memory_ltc))
    # print_mem_breakdown(mem_breakdown)
    # for node in ir_info:
    #     print(node)
if __name__ == "__main__":
    main()
