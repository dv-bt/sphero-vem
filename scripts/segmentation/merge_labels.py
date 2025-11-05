"""
Merge over-segmented neighboring labels
"""

from pathlib import Path
from tifffile import imread
from tqdm import tqdm
import numpy as np
from sphero_vem.postprocessing import (
    merge_labels,
    gaussian_edge_map,
    fill_internal_seams,
)
from sphero_vem.io import write_image, read_stack
from sphero_vem.utils import generate_manifest, get_seg_params


def main() -> None:
    stack_dir = Path(
        "data/processed/cropped/Au_01-vol_01/downscaled/downscaled-10-isotropic"
    )
    label_root = Path("data/processed/segmented/Au_01-vol_01")

    # Merge parameters
    sigma = 1
    merge_params_dict = {
        "cells": {
            "edge_thresh": 0.4,
            "rel_contact_thresh": 0.1,
            "sigma": sigma,
            "sphericity_thresh": 0.35,
        },
        "nuclei": {
            "edge_thresh": 0.15,
            "rel_contact_thresh": 0.1,
            "sigma": sigma,
            "sphericity_thresh": None,
        },
    }
    fill_params_dict = {"nuclei": {"close_radius": 2, "grow_dist": 3}}

    print("Reading image stack and computing edge map")
    image = read_stack(stack_dir, verbose=True)
    edge_map = gaussian_edge_map(image, sigma)

    label_dirs = list(label_root.glob("*/"))
    for label_dir in tqdm(label_dirs):
        try:
            # Get segmentation target
            seg_params = get_seg_params(label_dir)
            seg_target = seg_params.get("seg_target")
            if not seg_target:
                continue

            # Get merge parameters and continue if they don't match the target
            merge_params = merge_params_dict.get(seg_target)
            if not merge_params:
                continue

            label_path = label_dir / f"Au_01-vol_01-{seg_target}.tif"
            labels = imread(label_path)

            # Safety checks for labels
            assert labels.size > 0, "labels is empty"
            assert edge_map.size > 0, "edge_map is empty"
            assert labels.shape == edge_map.shape, "labels/edge_map shape mismatch"
            assert np.issubdtype(labels.dtype, np.integer), "labels must be integer"

            merged, _ = merge_labels(labels, edge_map=edge_map, **merge_params)
            processing = [{"step": "label_merging", **merge_params}]

            # Fill label seams if they're specified for the target
            fill_params = fill_params_dict.get(seg_target)
            if fill_params:
                merged = fill_internal_seams(merged, **fill_params)
                processing += [{"step": "fill_seams", **fill_params}]

            merged_dir = label_dir / "merged-labels"
            merged_dir.mkdir(exist_ok=True)
            generate_manifest(
                "Au_01-vol_01",
                merged_dir,
                [label_path],
                processing=processing,
            )
            write_image(merged_dir / label_path.name, merged, compressed=True)
        except (FileExistsError, FileNotFoundError, AssertionError):
            print(f"Errors for run {label_dir.name}")
            continue


if __name__ == "__main__":
    main()
