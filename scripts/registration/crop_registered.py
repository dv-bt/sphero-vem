"""
Crop registered images to exclude black borders
"""

from pathlib import Path
from tqdm import tqdm
import numpy as np
from tifffile import imread
from skimage import measure
from sphero_vem.io import write_image
from sphero_vem.utils import generate_manifest


def border_mask(image: np.ndarray, border_val: int = 0) -> np.ndarray:
    """
    Return a boolean mask of values that belong to components touching the image border.
    """
    zero = image == border_val
    labels = measure.label(zero, connectivity=1)
    H, W = image.shape[:2]

    touching = set()
    touching.update(np.unique(labels[0, :]))
    touching.update(np.unique(labels[H - 1, :]))
    touching.update(np.unique(labels[:, 0]))
    touching.update(np.unique(labels[:, W - 1]))

    touching.discard(0)
    if not touching:
        return np.zeros_like(zero, dtype=bool)

    mask = np.isin(labels, list(touching))
    return mask


def max_inscribed_rectangle(image: np.ndarray) -> tuple[int, int, int, int]:
    """
    Find maximum-area axis-aligned rectangle that avoids border-touching black pixels.

    The function uses an area under histogram approach with a monotonic stack algortithm.

    Parameters
    ----------
    image : np.ndarray
        The image to be cropped.

    Returns
    -------
    (top, bottom, left, right)
        Indices for slicing the image. Bottom and right are calculated to be exclusive.
    """
    border_zero = border_mask(image)
    valid = ~border_zero

    H, W = valid.shape
    heights = np.zeros(W, dtype=int)

    best_area = 0
    # (top, bottom, left, right)
    best_crop = (0, 0, 0, 0)

    for r in range(H):
        row = valid[r]
        heights = np.where(row, heights + 1, 0)
        stack = []
        c = 0
        while c <= W:
            h = heights[c] if c < W else 0
            if not stack or h >= heights[stack[-1]]:
                stack.append(c)
                c += 1
            else:
                top = stack.pop()
                height = heights[top]
                left_idx = stack[-1] + 1 if stack else 0
                width = c - left_idx
                area = height * width
                if area > best_area:
                    best_area = area
                    top_row = r - height + 1
                    bottom_row = r + 1
                    left_col = left_idx
                    right_col = left_idx + width
                    best_crop = (top_row, bottom_row, left_col, right_col)

    return best_crop


if __name__ == "__main__":
    root = Path("data/processed/aligned/Au_01-vol_01/")
    crop_dir = root / "cropped"
    crop_dir.mkdir(exist_ok=True, parents=True)

    image_list = sorted(root.glob("*.tif"))
    results = []
    for image_path in tqdm(image_list, "Analyzing images"):
        image = imread(image_path)
        results.append(max_inscribed_rectangle(image))
    crops = np.vstack(results)

    # Calculate the most restrictive crop and consider safety pixels to also include
    # potential artifacts from resampling
    safety_px = 1
    min_crop = (
        crops[:, 0].max() + safety_px,
        crops[:, 1].min() - safety_px,
        crops[:, 2].max() + safety_px,
        crops[:, 3].min() + safety_px,
    )
    for image_path in tqdm(image_list, "Cropping images"):
        image = imread(image_path)
        cropped = image[min_crop[0] : min_crop[1], min_crop[2] : min_crop[3]]
        write_image(crop_dir / image_path.name, cropped)

    generate_manifest(
        dataset=root.name,
        out_dir=crop_dir,
        images=image_list,
        processing=[{"step": "cropping", "crop": min_crop}],
    )
