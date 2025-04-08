import timeit
import torch
from functools import wraps

def timeit_decorator(number=1):
    """
    A decorator that uses timeit.timeit to measure the execution time of a function.
    
    This version expects the decorated function to receive a 'device' keyword argument.
    The decorator will extract the device and, if it is a GPU (CUDA or MPS), perform the necessary synchronization.
    
    Parameters:
        number (int): Number of times to execute the function for timing.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = None
            # Extract 'device' from the function's kwargs; default to CPU if not provided.
            device = kwargs.get("device", torch.device("cpu"))
            
            def call_func():
                nonlocal result
                result = func(*args, **kwargs)
                # When running torch functions, perform synchronization on GPU if needed.
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elif device.type == 'mps':
                    result.cpu()
            
            elapsed = timeit.timeit(call_func, number=number)
            avg_time = elapsed / number
            print(f"{func.__name__} executed {number} time(s) in {elapsed:.6f} seconds "
                  f"(avg: {avg_time:.6f} seconds per run)")
            return result
        return wrapper
    return decorator
