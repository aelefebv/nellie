import numpy as np
import torch


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
    # if its a list of tensors, convert to a single tensor
    elif isinstance(input_array, list):
        if isinstance(input_array[0], list):
            tensor_stack = []
            for tensor in input_array:
                tensor_stack.append(tensor[0])
            input_array = tensor_stack
        result = torch.stack(input_array)
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
            try:
                from_numpy = torch.from_numpy(input_array)
            except:
                from_numpy = torch.from_numpy(input_array.astype(np.float32))
                # logger.warning(f'failed to convert numpy array to tensor, converted to float32\n{input_array}')
            return from_numpy
        elif isinstance(input_array, list):
            if isinstance(input_array[0], list):
                tensor_stack = []
                for tensor in input_array:
                    tensor_stack.append(tensor[0])
                return torch.stack(tensor_stack)
            elif isinstance(input_array[0], torch.Tensor):
                return input_array
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


def zeros_like(input_array, dtype=None):
    input_array = to_tensor(input_array)
    if dtype == 'float64':
        dtype = torch.float64
    return torch.zeros_like(input_array, dtype=dtype)


def ones_like(input_array, dtype=None):
    input_array = to_tensor(input_array)
    if dtype == 'bool':
        dtype = torch.bool
    return torch.ones_like(input_array, dtype=dtype)


def ones(shape, dtype=None):
    if dtype == 'bool':
        dtype = torch.bool
    return torch.ones(shape, dtype=dtype)


def asarray(input_array, dtype=None):
    tensor_out = to_tensor(input_array)
    if dtype is not None:
        if dtype == 'double':
            dtype = torch.float64
        tensor_out = tensor_out.to(dtype)
    return tensor_out


def asnumpy(input_array):
    return input_array.cpu().numpy()


def gradient(input_array, axis=None):
    input_array = to_tensor(input_array)
    grad = list(torch.gradient(input_array, dim=axis))
    return grad


def max(input_array):
    input_array = to_tensor(input_array)
    return torch.max(input_array)


def min(input_array):
    input_array = to_tensor(input_array)
    return torch.min(input_array)


def abs(input_array):
    input_array = to_tensor(input_array)
    return torch.abs(input_array)


# def linalg(input_array, axis=None):
#     input_array = to_tensor(input_array)
#     return torch.linalg.norm(input_array, dim=axis)

def isinf(input_array):
    input_array = to_tensor(input_array)
    return torch.isinf(input_array)


def concatenate(input_array, axis=0):
    input_array = to_tensor(input_array)
    return torch.cat(input_array, dim=axis)


def argsort(input_array, axis=None):
    input_array = to_tensor(input_array)
    return torch.argsort(input_array, dim=axis)


def take_along_axis(input_array, indices, axis=None):
    input_array = to_tensor(input_array)
    return torch.take_along_dim(input_array, indices, dim=axis)


def exp(input_array):
    input_array = to_tensor(input_array)
    return torch.exp(input_array)


def nan_to_num(input_array, copy=True, nan=0.0):
    input_array = to_tensor(input_array)
    return torch.nan_to_num(input_array, nan=nan)


def where(condition, x, y):
    condition = to_tensor(condition)
    x = to_tensor(x)
    y = to_tensor(y)
    return torch.where(condition, x, y)


def percentile(input_array, q, axis=None):
    input_array = to_tensor(input_array)
    q /= 100
    return torch.quantile(input_array, q, dim=axis)


def inf():
    return torch.tensor(float('inf'))


def log10(input_array):
    input_array = to_tensor(input_array)
    return torch.log10(input_array)


def mean(input_array, axis=None):
    input_array = to_tensor(input_array)
    return torch.mean(input_array, dim=axis)
