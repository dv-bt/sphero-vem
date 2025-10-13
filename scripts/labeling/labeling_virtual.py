"""
Add virtual sections to labeled_01 and make new labeled_02 dataset.
Images are downscaled by a factor 5 to have isotropic voxels.
"""

from pathlib import Path
import numpy as np
import shutil
from sphero_vem.utils import read_manifest, generate_manifest
from sphero_vem.io import read_stack, read_tensor, write_image


def main():
    ds_factor = 5
    seed_y = 42
    seed_x = 1422
    excl = 150

    aligned_dir = Path("data/processed/aligned/Au_01-vol_01/downscaled/downscaled-5")
    labels_dir = Path("data/processed/labeled/Au_01-vol_01/labeled-01")
    out_dir = Path("data/processed/labeled/Au_01-vol_01/labeled-02")
    if out_dir.exists():
        usr = input(f"Directory {out_dir} already exists. Do you want to overwrite it?")
        if usr.lower() in ["yes", "y"]:
            shutil.rmtree(out_dir)
        else:
            print("Directory preserved. Script terminated")
            return
    out_dir.mkdir()

    processing_ds = [{"step": "downscaling", "factor": ds_factor}]
    processing_split = [
        {
            "step": "split",
            "random_seed_x": seed_x,
            "random_seed_y": seed_y,
            "excl": excl,
        }
    ]

    manifest_labels = read_manifest(labels_dir)
    manifest_aligned = read_manifest(aligned_dir)

    shutil.copytree(labels_dir, out_dir, dirs_exist_ok=True)
    # shutil.rmtree(out_dir / "bg-crops")
    # shutil.rmtree(out_dir / "np-crops")
    for image_path in out_dir.glob("*.tif"):
        image = read_tensor(
            image_path, dtype=None, ds_factor=ds_factor, return_4d=False
        )
        write_image(image_path, image.numpy(), compressed=False)

    for label_path in (out_dir / "labels").glob("*.tif"):
        labels = read_tensor(
            image_path,
            dtype=None,
            ds_factor=ds_factor,
            resample_mode="nearest",
            return_4d=False,
        )
        write_image(label_path, labels.numpy(), compressed=True)

    stack = read_stack(aligned_dir)

    # Exclude outer N slices due to potential black regions due to
    # padding during alignement
    virtual_paths = []

    # XZ slices
    rng = np.random.default_rng(seed=seed_y)
    subset_indices = rng.choice(stack.shape[1] - (2 * excl), size=20, replace=False)
    subset_indices += excl
    subset_indices.sort()

    for idx in subset_indices:
        slice = stack[:, idx, :]
        slice_name = f"Au_01-vol_01-y_{idx:04}-virt.tif"
        write_image(out_dir / slice_name, slice)
        virtual_paths.append(aligned_dir / slice_name)

    # YZ slices
    rng = np.random.default_rng(seed=seed_x)
    subset_indices = rng.choice(stack.shape[2] - (2 * excl), size=20, replace=False)
    subset_indices += excl
    subset_indices.sort()

    for idx in subset_indices:
        slice = stack[:, :, idx]
        slice_name = f"Au_01-vol_01-x_{idx:04}-virt.tif"
        write_image(out_dir / slice_name, slice)
        virtual_paths.append(aligned_dir / slice_name)

    images = [Path("data") / i for i in manifest_labels["inputs"]] + virtual_paths
    processing = processing_ds + [
        {"virtual sections": manifest_aligned["processing"] + processing_split}
    ]
    generate_manifest("Au_01-vol_01/labeled-02", out_dir, images, processing)


if __name__ == "__main__":
    main()
