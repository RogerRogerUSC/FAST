import torch


def top_k(tensor, k: int):
    values, indices = torch.topk(tensor.view(-1), k)
    sparse_tensor = torch.zeros_like(tensor)
    sparse_tensor.view(-1)[indices] = values
    return sparse_tensor


# To do: keep the random seed for two forward passes
def random_k(tensor, k: int):
    num_elements = tensor.numel()
    k = min(k, num_elements)
    non_zero_indices = torch.randperm(num_elements)[:k]
    sparse_tensor = torch.zeros_like(tensor)
    sparse_tensor.view(-1)[non_zero_indices] = tensor.view(-1)[non_zero_indices]
    return sparse_tensor


def quantize(tensor, num_bits=8):
    min_val, max_val = tensor.min(), tensor.max()
    qmin = 0
    qmax = 2**num_bits - 1
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale
    q_tensor = ((tensor / scale) + zero_point).round().clamp(qmin, qmax)
    q_tensor = q_tensor.to(torch.int32)
    return q_tensor, scale, zero_point


def dequantize_tensor(q_tensor, scale, zero_point):
    tensor = scale * (q_tensor - zero_point)
    return tensor
