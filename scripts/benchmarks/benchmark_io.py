from pathlib import Path
import numpy as np
import torch
import cv2
import imageio.v2 as imageio
from PIL import Image
import tifffile
from benchmark_timing import timeit_decorator

# ----------- Setup -----------
n_iters = 10
H, W = 10000, 10000
assets_path = Path("assets")

image_np = (np.random.rand(H, W) * 255).astype(np.uint8)
assets_path.mkdir(exist_ok=True)
tiff_path = assets_path / "benchmark_img.tiff"
tifffile.imwrite(tiff_path, image_np)

# Device detection and sync function
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# ----------- Read Benchmarks (with GPU transfer) -----------


@timeit_decorator(n_iters)
def read_cv2():
    img = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
    tensor = torch.from_numpy(img).to(device)
    return tensor

@timeit_decorator(n_iters)
def read_imageio():
    img = imageio.imread(tiff_path)
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor

@timeit_decorator(n_iters)
def read_pil():
    img = np.array(Image.open(tiff_path))
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor

@timeit_decorator(n_iters)
def read_tiff():
    img = tifffile.imread(tiff_path)
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor


# ----------- Write Benchmarks (CPU only) -----------
@timeit_decorator(n_iters)
def write_cv2():
    return cv2.imwrite(tiff_path, image_np)


@timeit_decorator(n_iters)
def write_imageio():
    return imageio.imwrite(tiff_path, image_np)


@timeit_decorator(n_iters)
def write_pil():
    return Image.fromarray(image_np).save(tiff_path)


@timeit_decorator(n_iters)
def write_tiff():
    return tifffile.imwrite(tiff_path, image_np)


# ----------- Run Benchmarks -----------

print(f"\n📥 Average READ times to GPU ({n_iters} runs) — device: {device}")
print("[OpenCV]")
read_cv2()
print("[imageio]")
read_imageio()
print("[PIL]")
read_pil()
print("[tifffile]")
read_tiff()

print(f"\n💾 Average WRITE times ({n_iters} runs)")
print("[OpenCV]")
write_cv2()
print("[imageio]")
write_imageio()
print("[PIL]")
write_pil()
print("[tifffile]")
write_tiff()
