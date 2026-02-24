"""
Get a downscaled version of a labeled dataset by downscaling the labels and indexing
the images from the resampled volume.
"""

from pathlib import Path
import json
import re
from tqdm import tqdm
import numpy as np
import zarr
from tifffile import imread
from skimage.transform import resize
from sphero_vem.io import write_image


def process_gt(path: Path, vol_shape: tuple[int, int, int]) -> dict:
    """Process ground truth image"""

    match = re.search(r"([x|y|z])_(\d+)-", path.name)
    axis = match.group(1)
    gt = imread(path)
    old_idx = int(match.group(2))
    idx = int(np.round(old_idx / 2))

    if "cells" in path.name:
        seg_target = "cells"
    elif "nuclei" in path.name:
        seg_target = "nuclei"
    else:
        raise ValueError("No seg target found")

    if axis == "x":
        indexer = np.s_[:, :, idx]
        shape = (vol_shape[0], vol_shape[1])
    elif axis == "y":
        indexer = np.s_[:, idx, :]
        shape = (vol_shape[0], vol_shape[2])
    elif axis == "z":
        indexer = np.s_[idx, :, :]
        shape = (vol_shape[1], vol_shape[2])

    new_image_name = f"Au_01-vol_01-{axis}_{idx:04}.tif"
    new_label_name = new_image_name.replace(".tif", f"-{seg_target}.tif")

    return {
        "gt": gt,
        "indexer": indexer,
        "new_image_name": new_image_name,
        "new_label_name": new_label_name,
        "old_name": path.name,
        "shape": shape,
        "seg_target": seg_target,
    }


def main():
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")
    label_root = Path("data/processed/labeled/Au_01-vol_01/labeled-06/")
    src_dir = label_root / "50-50-50/labels"
    dst_dir = label_root / "100-100-100"

    root = zarr.open_group(root_path, mode="r")
    images = root.get("images/100-100-100")

    gts = [process_gt(path, images.shape) for path in sorted(src_dir.glob("*.tif"))]

    for item in tqdm(gts, "Writing downsampled labels"):
        new_image_path = dst_dir / f"{item['new_image_name']}"
        new_label_path = dst_dir / f"labels/{item['new_label_name']}"
        new_label_path.parent.mkdir(exist_ok=True, parents=True)

        image = images[item["indexer"]]
        labels = resize(
            item["gt"],
            output_shape=item["shape"],
            order=0,
        )
        assert image.shape == labels.shape

        write_image(new_image_path, image, compressed=True)
        write_image(new_label_path, labels, compressed=True)

    with open(dst_dir / "manifest.json", "w") as file:
        manifest = images.attrs.asdict()
        manifest["orig_labels"] = [i["old_name"] for i in gts]
        manifest["new_labels"] = [i["new_label_name"] for i in gts]
        json.dump(manifest, file, indent=4)


if __name__ == "__main__":
    main()
