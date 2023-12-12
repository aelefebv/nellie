import torch
import numpy as np

def flip(input_array, dims):
    if not isinstance(input_array, torch.Tensor):
        input_array = torch.tensor(input_array)
    return torch.flip(input_array, dims=dims)

def array(input_array, dtype=None, device=None):
    # Convert input to a PyTorch tensor
    if isinstance(input_array, torch.Tensor):
        result = input_array.clone()
    elif isinstance(input_array, np.ndarray):
        result = torch.from_numpy(input_array)
    else:
        result = torch.tensor(input_array)

    # Handle dtype
    if dtype is not None:
        result = result.to(dtype)

    # Handle device
    if device is not None:
        result = result.to(device)

    return result

def to_tensor(input_array):
    if not isinstance(input_array, torch.Tensor):
        if isinstance(input_array, np.ndarray):
            return torch.from_numpy(input_array)
        else:
            return torch.tensor(input_array)
    return input_array

def var(input_array):
    # make sure input is a float tensor
    input_array = to_tensor(input_array).float()
    return torch.var(input_array, unbiased=False)

def histogram(input_array, bins, range):
    input_array = to_tensor(input_array).float()
    hist = torch.histc(input_array, bins=bins, min=range[0], max=range[1])
    bin_edges = torch.linspace(range[0], range[1], steps=bins + 1)
    return hist, bin_edges

def sum(input_array):
    input_array = to_tensor(input_array)
    return torch.sum(input_array)

def cumsum(input_array):
    input_array = to_tensor(input_array)
    return torch.cumsum(input_array, dim=0)

def argmax(input_array):
    input_array = to_tensor(input_array)
    return torch.argmax(input_array, dim=0)

def flatnonzero(input_array):
    input_array = to_tensor(input_array)
    return torch.nonzero(input_array.flatten(), as_tuple=False).flatten()

def arange(start, end=None, step=1):
    # Note: arange typically doesn't need conversion to tensor for its arguments
    if end is None:
        return torch.arange(start)
    else:
        return torch.arange(start, end, step)

def sqrt(input_array):
    input_array = to_tensor(input_array)
    return torch.sqrt(input_array)
