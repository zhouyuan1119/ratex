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
from ratex.utils.mem_model_utils import analyze_training_peak_memory, profile_training_peak_memory
from transformers import GPT2Config, GPT2ForSequenceClassification

class GPT2ModelWrapper(nn.Module):
    """
        A wrapper around huggingface GPT2Model for memory estimation with our APIs. 
        The wrapped model only returns the "pooled outputs" from BertModel. 
    """
    def __init__(self, config):
        super(GPT2ModelWrapper, self).__init__()
        self.num_labels = config.num_labels
        self.gpt = GPT2ForSequenceClassification(config)
    
    def forward(self, x, **kwargs):
        return self.gpt.forward(x, return_dict=False, **kwargs)[0].view(-1, self.num_labels)

GPT2Config()
config = GPT2Config(
    vocab_size=50257,
    n_embd=768,
    n_layer=12,
    n_head=12,
    # activation_function="relu",
    num_labels=10,
    pad_token_id=0
)

""" Missing ops: 
    - index
"""

def main():
    model_lt = GPT2ModelWrapper(config)
    model_cuda = copy.deepcopy(model_lt)
    model_lt = model_lt.to(device="lazy")
    optimizer_lt = torch.optim.SGD(model_lt.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    peak_memory_mbs = analyze_training_peak_memory(
        model_lt, optimizer_lt, loss_fn, 
        input_shape=(8, 512), output_shape=(8,), 
        input_dtype=torch.int64, output_dtype=torch.int64, 
        input_range=[0, 50257], output_range=[0, 10],
        n_batches=2)
    model_cuda = model_cuda.cuda()
    optimizer = optim.SGD(model_cuda.parameters(), lr=0.001)
    peak_memory_bs = profile_training_peak_memory(
        model_cuda, optimizer, torch.nn.NLLLoss(), 
        input_shape=(8, 512), output_shape=(8,), 
        input_dtype=torch.int64, output_dtype=torch.int64, 
        input_range=[0, 50257], output_range=[0, 10],
        n_batches=2)

if __name__ == "__main__":
    main()
