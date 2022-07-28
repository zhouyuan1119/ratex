# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

""" Utilities for PyTorch memory modeling """
import torch
import ratex.lazy_tensor_core.core.lazy_model as lm

def random_torch_tensor(shape, dtype, range=None):
    """ Simple utility to create a random torch tensor of given type """
    if dtype == torch.float32:
        assert range is None, 'Float with range not supported!'
        return torch.randn(*shape, dtype=dtype)
    elif dtype == torch.int64:
        if range:
            return torch.randint(range[0], range[1], shape, dtype=dtype)
        else:
            return torch.zeros(*shape, dtype=dtype)
    else:
        raise NotImplementedError

def analyze_training_peak_memory(model, optimizer, loss_fn, input_shape, output_shape, 
                                 input_dtype, output_dtype, output_range=None, n_batches=2):
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
        output_range: Range of output (for classification labels). 
    
    Returns:
        peak_mem_mbs: Peak memory consumption in MBs. 
    """
    
    for _, p in model.named_parameters():
        assert p.device.type == 'lazy', 'The model is not on the lazy device, please run model.to(device="lazy") first. '

    peak_mem_mbs = -float('inf')
    for batch in range(n_batches):
        # Create dummy inputs
        inputs = random_torch_tensor(input_shape, input_dtype)
        inputs = inputs.to(device="lazy")
        inputs.requires_grad = True
        labels = random_torch_tensor(output_shape, output_dtype, output_range)
        labels = labels.to(device="lazy")  # One-hot
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        lm.mark_step()
        peak_mem_batch = lm.get_peak_memory()
        print('Analyzed peak memory for batch {}: {}'.format(batch, peak_mem_batch))
        peak_mem_mbs = max(peak_mem_mbs, peak_mem_batch)
    
    return peak_mem_mbs

def profile_training_peak_memory(model, optimizer, loss_fn, input_shape, output_shape, 
                                 input_dtype, output_dtype, output_range, n_batches=2):
    """Same with the function above, except that the peak memory is retrived from PyTorch CUDA utils."""

    for _, p in model.named_parameters():
        assert p.device.type == 'cuda', 'The model is not on GPU, please run model.to(device="cuda") first. '

    for i in range(n_batches):
        torch.cuda.reset_max_memory_allocated()
        # Create dummy inputs
        inputs = random_torch_tensor(input_shape, input_dtype)
        inputs = inputs.cuda()
        inputs.requires_grad = True
        labels = random_torch_tensor(output_shape, output_dtype, output_range)
        labels = labels.cuda()  # One-hot
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        print("Profiled peak memory for batch {}: {}".format(i, torch.cuda.max_memory_allocated() / (1024*1024)))
    return torch.cuda.max_memory_allocated()
