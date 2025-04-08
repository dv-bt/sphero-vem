"""
Compare denoising performance of different libraries.
Using the non-local means denoising algorithm
DOI:10.1109/CVPR.2005.38
"""

from pathlib import Path
import time
import tifffile
from tqdm import tqdm
import cv2
from skimage.restoration import denoise_nl_means

# Optional: Set a large EM-like image size
# H, W = 10000, 10000
# image_np = np.random.rand(H, W).astype(np.float32)

# Use a real image
image_path = Path("../../data/raw/Au_01-vol_01/Au_01-vol_01-z_0900.tif")
image_np = tifffile.imread(image_path)

# -------------------
# skimage resize
# -------------------
start = time.time()
for _ in tqdm(range(5), "Denoising with scikit-image: "):
    out_skimage = denoise_nl_means(image_np)
print(f"[skimage]  Time per call: {(time.time() - start) / 10:.5f} s")

# -------------------
# OpenCV resize
# -------------------
start = time.time()
for _ in tqdm(range(5), "Denoising with open-cv: "):
    out_cv2 = cv2.fastNlMeansDenoising(image_np)
print(f"[OpenCV]   Time per call: {(time.time() - start) / 10:.5f} s")
