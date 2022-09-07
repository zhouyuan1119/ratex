# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import torch
import ratex
import _RATEXC
_RATEXC._set_ratex_vlog_level(-5)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from ratex.utils.mem_model_utils import analyze_training_peak_memory, profile_training_peak_memory, print_mem_breakdown, wrap_model
from transformers import BertForSequenceClassification, BertConfig

class BertModelWrapper(nn.Module):
    """
        A wrapper around huggingface BertModel for memory estimation with our APIs. 
        The wrapped model only returns the "pooled outputs" from BertModel. 
    """
    def __init__(self, config):
        super(BertModelWrapper, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertForSequenceClassification(config)
    
    def forward(self, x, **kwargs):
        res = self.bert(x, return_dict=False, **kwargs)[0].view(-1, self.num_labels)
        return res


config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=2,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="relu",
    num_labels=10)

def main():
    model_lt = BertModelWrapper(config)
    model_cuda = copy.deepcopy(model_lt)
    # model_lt = wrap_model(model_lt)
    model_lt = model_lt.to(device="lazy")
    optimizer_lt = torch.optim.SGD(model_lt.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    peak_memory_ltc, mem_breakdown = analyze_training_peak_memory(
        model_lt, optimizer_lt, loss_fn, 
        input_shape=(64, 256), output_shape=(64,), 
        input_dtype=torch.int64, output_dtype=torch.int64, 
        input_range=[0, 30522], output_range=[0, 10],
        n_batches=2)
    model_cuda = model_cuda.cuda()
    optimizer = optim.SGD(model_cuda.parameters(), lr=0.001)
    peak_memory_profiled = profile_training_peak_memory(
        model_cuda, optimizer, torch.nn.NLLLoss(), 
        input_shape=(64, 256), output_shape=(64,), 
        input_dtype=torch.int64, output_dtype=torch.int64, 
        input_range=[0, 30522], output_range=[0, 10],
        n_batches=2)
    print('Profiled peak memory: {0:6.2f} MBs'.format(peak_memory_profiled))
    print('Analyzed peak memory: {0:6.2f} MBs'.format(peak_memory_ltc))
    print_mem_breakdown(mem_breakdown)

if __name__ == "__main__":
    main()
