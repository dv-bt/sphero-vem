"""
Convert labels to use the aligned raw images.
"""

import re
import json
from pathlib import Path
import numpy as np
import zarr
from tifffile import imread
from sphero_vem.io import write_image
from skimage.transform import resize


def process_gt(
    path: Path, crop: tuple[int, int, int, int], inputs_map: dict[int, int]
) -> dict:
    """Process ground truth image"""
    y_start, y_stop, x_start, x_stop = crop

    match = re.search(r"([x|y|z])_(\d+)-", path.name)
    axis = match.group(1)
    idx = int(match.group(2))
    gt = imread(path)
    virt = True if "virt" in path.name else False

    if "cells" in path.name:
        seg_target = "cells"
    elif "nuclei" in path.name:
        seg_target = "nuclei"
    else:
        raise ValueError("No seg target found")

    if axis == "x":
        gt = gt[:, y_start:y_stop]
        idx -= x_start
        indexer = np.s_[:, :, idx]
    elif axis == "y":
        gt = gt[:, x_start:x_stop]
        idx -= y_start
        indexer = np.s_[:, idx, :]
    elif axis == "z":
        idx = inputs_map[idx]
        gt = gt[y_start:y_stop, x_start:x_stop]
        indexer = np.s_[idx, :, :]

    new_image_name = f"Au_01-vol_01-{axis}_{idx:04}.tif"
    new_label_name = new_image_name.replace(".tif", f"-{seg_target}.tif")

    return {
        "gt": gt,
        "indexer": indexer,
        "new_image_name": new_image_name,
        "new_label_name": new_label_name,
        "old_path": path,
        "virt": virt,
        "seg_target": seg_target,
    }


def main():
    ## Data and save paths
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")
    gt_root = Path("data/processed/labeled/Au_01-vol_01/labeled-02/labels")

    ## Load data and calculate crop parameters
    root = zarr.open_group(root_path)
    images_fullres = root.get("images/50-10-10")
    images = root.get("images/50-50-50")
    gt_paths = list(gt_root.glob("*.tif"))

    crop_fullres = images.attrs.get("processing")[-2]["crop"]
    scale_factors = images.attrs.get("processing")[-1]["scale_factors"]
    factors = [scale_factors[1]] * 2 + [scale_factors[2]] * 2
    crop = [int(idx / factors[i]) for i, idx in enumerate(crop_fullres)]

    inputs = images_fullres.attrs.get("inputs")
    inputs_map = {
        int(re.search(r"-z_(\d+)", input).group(1)): i for i, input in enumerate(inputs)
    }

    gts = [process_gt(i, crop=crop, inputs_map=inputs_map) for i in gt_paths]

    ## Process spacing (50-50-50)
    new_label_root = Path("data/processed/labeled/Au_01-vol_01/labeled-04/50-50-50")

    for item in gts:
        virt_subdir = "virt/" if item["virt"] else ""
        new_image_path = new_label_root / f"{virt_subdir}{item['new_image_name']}"
        new_label_path = (
            new_label_root / f"{virt_subdir}labels/{item['new_label_name']}"
        )
        new_label_path.parent.mkdir(exist_ok=True, parents=True)

        image = images[item["indexer"]]
        labels = item["gt"]
        assert image.shape == labels.shape

        write_image(new_image_path, image, compressed=True)
        write_image(new_label_path, labels, compressed=True)

    with open(new_label_root / "manifest.json", "w") as file:
        json.dump(images.attrs.asdict(), file, indent=4)

    ## Processs coarser spacing (50-100-100)
    images_lowres = root.get("images/50-100-100")
    new_label_root = Path("data/processed/labeled/Au_01-vol_01/labeled-04/50-100-100")

    for item in gts:
        if item["virt"]:
            continue
        new_image_path = new_label_root / f"{item['new_image_name']}"
        new_label_path = new_label_root / f"labels/{item['new_label_name']}"
        new_label_path.parent.mkdir(exist_ok=True, parents=True)

        image = images_lowres[item["indexer"]]
        labels = resize(
            item["gt"],
            output_shape=images_lowres.shape[1:],
            order=0,
        )
        assert image.shape == labels.shape

        write_image(new_image_path, image, compressed=True)
        write_image(new_label_path, labels, compressed=True)

    with open(new_label_root / "manifest.json", "w") as file:
        json.dump(images_lowres.attrs.asdict(), file, indent=4)


if __name__ == "__main__":
    main()
