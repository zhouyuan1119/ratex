# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

""" Utilities for PyTorch memory modeling """
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
        return self.dummy(res)

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
        print('  |-fwd: peak {0:6.2f}, param {1:6.2f}, input_act {2:6.2f}, output_act {3:6.2f}, isolated {4:6.2f}'.format(
          info['fwd']['peak_mem'], info['fwd']['param'], info['fwd']['input_act'], info['fwd']['output_act'], info['fwd']['peak_mem_isolated']))
        print('  |-bwd: peak {0:6.2f}, param {1:6.2f}, input_act {2:6.2f}, output_act {3:6.2f}, isolated {4:6.2f}'.format(
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
        # Create dummy inputs
        inputs = random_torch_tensor(input_shape, input_dtype, input_range)
        inputs = inputs.to(device="lazy")
        labels = random_torch_tensor(output_shape, output_dtype, output_range)
        labels = labels.to(device="lazy")  # One-hot
        marker_tensor = random_torch_tensor((1,), torch.float32)
        marker_tensor = marker_tensor.to(device="lazy")
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        # Insert an additional dummy op here to mark the boundary of the last layer's bwd
        _ = DummyFunc.apply(marker_tensor)
        optimizer.step()
        lm.mark_step()
        peak_mem_batch = lm.get_peak_memory()
        peak_mem_mbs = max(peak_mem_mbs, peak_mem_batch)
        if batch != n_batches - 1:
            LayerWrapper.executed_layers.clear()
    
    mem_breakdown = lm.get_memory_breakdown()
    num_all_info = len(mem_breakdown)
    breakdown_dict = dict()
    print(LayerWrapper.executed_layers)
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
        # inputs.requires_grad = True
        labels = random_torch_tensor(output_shape, output_dtype, output_range)
        labels = labels.cuda()  # One-hot
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    return torch.cuda.max_memory_allocated() / (1024*1024)

def wrap_model(model: torch.nn.Module, cur_depth: int = 0, max_depth: int = 1):
    """
        Wrap children of the original model with the layer wrapper above.

        Inputs:
        - model: the original model
        - cur_depth: current depth, used for recursion
        - max_depth: this function can operate recursively if depth > 1, where children
          of children will also be wrapped

        Returns: the wrapped model
    """

    for name in dir(model):
        member = getattr(model, name)
        if isinstance(member, torch.nn.Module):
            wrapped_member = LayerWrapper(member, name)
            setattr(model, name, wrapped_member)
    return model
