"""
Merge over-segmented neighboring labels
"""

from pathlib import Path
from tifffile import imread
from tqdm import tqdm
import scipy.ndimage as ndi
import numpy as np
from sphero_vem.postprocessing import merge_labels
from sphero_vem.io import write_image, read_stack
from sphero_vem.utils import generate_manifest


def compute_edges(image: np.ndarray, sigma: int) -> np.ndarray:
    edge_map = ndi.gaussian_gradient_magnitude(image, sigma, np.float32)
    p1, p99 = np.percentile(edge_map, (1, 99))
    edge_map = np.clip((edge_map - p1) / (p99 - p1), 0, 1)
    return edge_map


def main() -> None:
    stack_dir = Path("data/processed/aligned/Au_01-vol_01/downscaled/downscaled-2")
    label_root = Path("data/processed/segmented/Au_01-vol_01")

    # Merge parameters
    merge_params = {
        "edge_thresh": 0.15,
        "rel_contact_thresh": 0.1,
        "sigma": 1,
        "connectivity": 2,
    }

    print("Reading image stack and computing edge map")
    image = read_stack(stack_dir, verbose=False)
    edge_map = compute_edges(image, merge_params["sigma"])

    label_dirs = list(label_root.glob("*nuclei*/"))
    for label_dir in tqdm(label_dirs):
        try:
            label_path = label_dir / "Au_01-vol_01-nuclei.tif"
            labels = imread(label_path)

            merged, _ = merge_labels(labels, edge_map=edge_map, **merge_params)

            merged_dir = label_dir / "merged-labels"
            merged_dir.mkdir(exist_ok=True)
            generate_manifest(
                "Au_01-vol_01",
                merged_dir,
                [label_path],
                processing=[{"step": "label_merging", **merge_params}],
            )
            write_image(merged_dir / label_path.name, merged, compressed=True)
        except FileExistsError:
            continue


if __name__ == "__main__":
    main()
