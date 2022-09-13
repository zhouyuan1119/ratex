# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

""" Utilities for PyTorch memory modeling """
from typing import Iterable
import torch
import ratex.lazy_tensor_core.core.lazy_model as lm
from ratex.core.lazy_model import dummy

class DummyFunc(torch.autograd.Function):
    """
        A function to insert a dummy op into the LTC IR. 
    """

    @staticmethod
    def forward(ctx, input):
        # We don't need to save anything
        return dummy(input)
    
    @staticmethod
    def backward(ctx, grad_output):
        return dummy(grad_output)

class LayerWrapper(torch.nn.Module):
    """
        A simple wrapper around a PyTorch NN module. This wrapper will insert one special op after
        the forward/backward pass of the wrapped module. 
    """

    # A list shared by all wrappers to track how layers are executed
    executed_layers = []

    def __init__(self, mod: torch.nn.Module, name: str = None):
        super(LayerWrapper, self).__init__()
        self.name_ = name
        self.mod_ = mod
        self.dummy = DummyFunc.apply

    def forward(self, *args, **kwargs):
        res = self.mod_(*args, **kwargs)
        LayerWrapper.executed_layers.append(self.name_)
        # Custom autograd function must take a single tensor as input
        # https://github.com/pytorch/pytorch/issues/55509#issuecomment-815160271
        if isinstance(res, torch.Tensor):
            ret = self.dummy(res)
        else:
            assert isinstance(res, Iterable), "Result must be single tensor or iterable!"
            ret = tuple(self.dummy(elm) if isinstance(elm, torch.Tensor) else elm for elm in res)
        return ret

def random_torch_tensor(shape, dtype, range=None):
    """ Simple utility to create a random torch tensor of given type """
    if (dtype == torch.float32) or (dtype == torch.float16) or (dtype == torch.float64):
        assert range is None, 'Float with range not supported!'
        return torch.randn(*shape, dtype=dtype)
    elif dtype == torch.int64:
        if range:
            return torch.randint(range[0], range[1], shape, dtype=dtype)
        else:
            return torch.zeros(*shape, dtype=dtype)
    else:
        raise NotImplementedError

def print_mem_breakdown(mem_breakdown):
    """
        Print the memory breakdown. 
    """
    print('Analyzed memory breakdown:')
    for layer_name, info in mem_breakdown.items():
        print('|-{}:'.format(layer_name))
        print('  |-fwd: peak {0:6.4f}, param {1:6.4f}, input_act {2:6.4f}, output_act {3:6.4f}, isolated {4:6.4f}'.format(
          info['fwd']['peak_mem'], info['fwd']['param'], info['fwd']['input_act'], info['fwd']['output_act'], info['fwd']['peak_mem_isolated']))
        print('  |-bwd: peak {0:6.4f}, param {1:6.4f}, input_act {2:6.4f}, output_act {3:6.4f}, isolated {4:6.4f}'.format(
          info['bwd']['peak_mem'], info['bwd']['param'], info['bwd']['input_act'], info['bwd']['output_act'], info['bwd']['peak_mem_isolated']))

def analyze_training_peak_memory(model, optimizer, loss_fn, input_shape, output_shape, 
                                 input_dtype, output_dtype, input_range=None, output_range=None, n_batches=2):
    """
    Get the peak memory consumption while training the model using an analysis
    pass on the lazy tensor IR. Assuming that the model is in device "lazy". 

    Args:
        model: torch.nn.Module  The model to be analyzed. 
        optimizer: torch.optim.Optimizer  The optimizer to be used during training. 
        loss_fn: torch.nn.Module  The loss function. 
        input_shape: Tuple[int]  Input shape, including the batch dimension. Used to construct random input. 
        output_shape: Tuple[int]  Output shape, including the batch dimension. Used to construct random labels. 
        input_dtype: Data type of the input. 
        output_dtype: Data type of the output. 
        input_range: Range of input (for categorical features, like words). 
        output_range: Range of output (for classification labels). 
    
    Returns:
        peak_mem_mbs: Peak memory consumption in MBs. 
    """
    
    for _, p in model.named_parameters():
        assert p.device.type == 'lazy', 'The model is not on the lazy device, please run model.to(device="lazy") first. '

    peak_mem_mbs = -float('inf')
    for batch in range(n_batches):
        print('Start batch {}!'.format(batch))
        # Create dummy inputs
        inputs = random_torch_tensor(input_shape, input_dtype, input_range)
        inputs = inputs.to(device="lazy")
        labels = random_torch_tensor(output_shape, output_dtype, output_range)
        labels = labels.to(device="lazy")  # One-hot
        optimizer.zero_grad()
        outputs = model(inputs)
        print('Executed layers: ', LayerWrapper.executed_layers)
        loss = loss_fn(outputs, labels)
        loss.backward()
        # Insert an additional dummy op here to mark the boundary of the last layer's bwd
        marker_tensor = random_torch_tensor((1,), torch.float32)
        marker_tensor = marker_tensor.to(device="lazy")
        _ = DummyFunc.apply(marker_tensor)
        optimizer.step()
        lm.mark_step()
        print('Done batch {}!'.format(batch))
        peak_mem_batch = lm.get_peak_memory()
        peak_mem_mbs = max(peak_mem_mbs, peak_mem_batch)
        if batch != n_batches - 1:
            LayerWrapper.executed_layers.clear()
    
    mem_breakdown = lm.get_memory_breakdown()
    num_all_info = len(mem_breakdown)
    print('num_all_info = ', num_all_info)
    breakdown_dict = dict()
    for idx, layer_name in enumerate(LayerWrapper.executed_layers):
        print(idx, layer_name)
        fwd_info = mem_breakdown[idx]
        bwd_info = mem_breakdown[num_all_info-idx-1]
        breakdown_dict[layer_name] = {
            'fwd': fwd_info,
            'bwd': bwd_info
        }

    return peak_mem_mbs, breakdown_dict

def profile_training_peak_memory(model, optimizer, loss_fn, input_shape, output_shape, 
                                 input_dtype, output_dtype, input_range=None, output_range=None, n_batches=2):
    """Same with the function above, except that the peak memory is retrived from PyTorch CUDA utils."""

    for _, p in model.named_parameters():
        assert p.device.type == 'cuda', 'The model is not on GPU, please run model.to(device="cuda") first. '

    for i in range(n_batches):
        torch.cuda.reset_max_memory_allocated()
        # Create dummy inputs
        inputs = random_torch_tensor(input_shape, input_dtype, input_range)
        inputs = inputs.cuda()
        labels = random_torch_tensor(output_shape, output_dtype, output_range)
        labels = labels.cuda()  # One-hot
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    return torch.cuda.max_memory_allocated() / (1024*1024)

def wrap_model(model: torch.nn.Module, name: str, cur_depth: int = 0, max_depth: int = 1):
    """
        Wrap children of the original model with the layer wrapper above.

        Inputs:
        - model: the original model
        - name: the name of the model (layer)
        - cur_depth: current depth, used for recursion
        - max_depth: this function can operate recursively if depth > 1, where children
          of children will also be wrapped

        Returns: the wrapped model
    """
    # Base case: when reaching max depth, don't go inside and wrap the layer as a whole
    if cur_depth >= max_depth:
        return LayerWrapper(model, name)

    has_valid_children = False
    for layer_name in dir(model):
        # Skip properties
        try:
            member_from_type = getattr(type(model), layer_name)
        except Exception:
            member_from_type = None
        if isinstance(member_from_type, property):
            continue

        member = getattr(model, layer_name)
        # For PyTorch module list, wrap each element instead. 
        # We consider this as going in one more layer. 
        if isinstance(member, torch.nn.ModuleList):
            has_valid_children = True
            new_elms = []
            for idx, elm in enumerate(member):
                if isinstance(elm, torch.nn.Module):
                    elm_name = layer_name + '_{}'.format(idx)
                    wrapped_elm = wrap_model(elm, elm_name, cur_depth+1, max_depth)
                    new_elms.append(wrapped_elm)
                    print('[wrap_model] Depth {}: wrap {}'.format(cur_depth, elm_name))
            setattr(model, layer_name, torch.nn.ModuleList(new_elms))
        elif isinstance(member, torch.nn.Module):
            has_valid_children = True
            wrapped_member = wrap_model(member, layer_name, cur_depth+1, max_depth)
            setattr(model, layer_name, wrapped_member)
            print('[wrap_model] Depth {}: wrap {}'.format(cur_depth, layer_name))
    return model if has_valid_children else LayerWrapper(model, name)
