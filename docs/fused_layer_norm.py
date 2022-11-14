# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This code is copied fron NVIDIA apex:
      https://github.com/NVIDIA/apex
   with some changes. """

import torch
import ratex
import ratex.lazy_tensor_core.core.lazy_model as lm

from ratex.utils.mem_model_utils import FusedLayerNormAffineFunc

bs = 32
normalized_shape = (1024, 1024)
elementwise_affine = True
eps = 1e-5

if __name__ == "__main__":
   input = torch.rand((bs, ) + normalized_shape, dtype=torch.float32)
   input = input.detach().requires_grad_(True)
   input.retain_grad()
   input = input.to(device="lazy")
   weight = torch.rand(normalized_shape, dtype=torch.float32)
   weight = weight.detach().requires_grad_(True)
   weight.retain_grad()
   weight = weight.to(device="lazy")
   bias = torch.rand(normalized_shape, dtype=torch.float32)
   bias = bias.detach().requires_grad_(True)
   bias.retain_grad()
   bias = bias.to(device="lazy")
   output = FusedLayerNormAffineFunc.apply(input, weight, bias, normalized_shape, eps)
   fake_label = torch.rand_like(output)
   print(output.requires_grad)
   loss = (fake_label - output).sum()
   loss.backward()
   lm.mark_step()
