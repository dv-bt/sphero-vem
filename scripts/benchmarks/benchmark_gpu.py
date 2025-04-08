"""
Benchmark torch GPU (Cuda/Metal) performance
"""

import torch
from benchmark_timing import timeit_decorator


n_iters = 5  # Number of iterations for timing
size = 10000  # Matrix dimension (creates size x size matrices)


@timeit_decorator(n_iters)
def benchmark(device):
    # Create two random matrices on the specified device
    x = torch.randn(size, size, device=device)
    y = torch.randn(size, size, device=device)
    return torch.mm(x, y)


print("Executing benchmark on CPU...")
benchmark(device=torch.device("cpu"))
if torch.cuda.is_available():
    gpu_device = torch.device("cuda")
elif torch.mps.is_available():
    gpu_device = torch.device("mps")
else:
    gpu_device = None
if gpu_device:
    print(f"Executing benchmark on GPU ({gpu_device})...")
    benchmark(device=gpu_device)
else:
    print("GPU not detected, skipping benchmark")
