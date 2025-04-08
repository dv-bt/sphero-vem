"""
Compare downscaling performance of different libraries
"""

import time
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
from skimage.transform import resize

# Optional: Set a large EM-like image size
H, W = 10000, 10000
image_np = np.random.rand(H, W).astype(np.float32)

# Torch tensor version
image_torch = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

# Check for GPU
# Use MPS
device = torch.device("cpu")
image_torch = image_torch.to(device)

# Target size (downscale by factor of 2)
target_size = (H // 2, W // 2)

# -------------------
# Torch interpolate
# -------------------
start = time.time()
# Consder conversion overhead
for _ in range(10):
    image_torch = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    out_torch = F.interpolate(
        image_torch, size=target_size, mode="bilinear", antialias=True
    )
    image_np = image_torch.squeeze().numpy()
# torch.mps.synchronize()
print(f"[Torch]        Time per call: {(time.time() - start) / 10:.5f} s")


# -------------------
# Torchvision interpolate
# -------------------
start = time.time()
# Consder conversion overhead
for _ in range(10):
    image_torch = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    image_torch.to("cpu")  # Actually faster on CPU than MPS? Should be tested on cuda
    out_torchvision = transforms.Resize(target_size)(image_torch)
    image_np = image_torch.squeeze().numpy()
# torch.mps.synchronize()
print(f"[Torchvision]  Time per call: {(time.time() - start) / 10:.5f} s")

# -------------------
# skimage resize
# -------------------
start = time.time()
for _ in range(10):
    out_skimage = resize(
        image_np, target_size, order=1, mode="reflect", anti_aliasing=True
    )
print(f"[skimage]      Time per call: {(time.time() - start) / 10:.5f} s")

# -------------------
# OpenCV resize
# -------------------
start = time.time()
for _ in range(10):
    out_cv2 = cv2.resize(
        image_np, dsize=(target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR
    )
print(f"[OpenCV]       Time per call: {(time.time() - start) / 10:.5f} s")
