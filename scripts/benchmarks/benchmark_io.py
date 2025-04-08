import time
import numpy as np
import torch
import cv2
import imageio.v2 as imageio
from PIL import Image
import tifffile

# ----------- Setup -----------
H, W = 10000, 10000
image_np = (np.random.rand(H, W) * 255).astype(np.uint8)
tiff_path = "benchmark_img.tiff"
tifffile.imwrite(tiff_path, image_np)

# Device detection and sync function
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


# ----------- Timing Helper -----------
def timeit(fn, n=10, sync=True):
    start = time.time()
    for _ in range(n):
        fn()
        if sync:
            sync_device()
    return (time.time() - start) / n


# ----------- Read Benchmarks (with GPU transfer) -----------


def read_cv2():
    img = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
    tensor = torch.from_numpy(img).to(device)
    return tensor


def read_imageio():
    img = imageio.imread(tiff_path)
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor


def read_pil():
    img = np.array(Image.open(tiff_path))
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor


def read_tiff():
    img = tifffile.imread(tiff_path)
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor


# ----------- Write Benchmarks (CPU only) -----------
def write_cv2():
    return cv2.imwrite("out_cv2.png", image_np)


def write_imageio():
    return imageio.imwrite("out_imageio.png", image_np)


def write_pil():
    return Image.fromarray(image_np).save("out_pil.png")


def write_tiff():
    return tifffile.imwrite("out_tiff.tif", image_np)


# ----------- Run Benchmarks -----------

print(f"\n📥 Average READ times to GPU (10 runs) — device: {device}")
print(f"[OpenCV]      {timeit(read_cv2):.5f} s")
print(f"[imageio]     {timeit(read_imageio):.5f} s")
print(f"[PIL]         {timeit(read_pil):.5f} s")
print(f"[tifffile]    {timeit(read_tiff):.5f} s")

print("\n💾 Average WRITE times (10 runs)")
print(f"[OpenCV]      {timeit(write_cv2, sync=False):.5f} s")
print(f"[imageio]     {timeit(write_imageio, sync=False):.5f} s")
print(f"[PIL]         {timeit(write_pil, sync=False):.5f} s")
print(f"[tifffile]    {timeit(write_tiff, sync=False):.5f} s")
