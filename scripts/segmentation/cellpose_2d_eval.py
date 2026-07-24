"""
Evaluation of segmentation masks of 2D datasets segmented with both pretrained and
finetuned cellpose models.
"""

from pathlib import Path
from typing import NamedTuple
from itertools import product

from tqdm import tqdm
from sphero_vem.segmentation.cellpose import evaluate_segmentation


class LabelDataset(NamedTuple):
    name: str
    scale: str


def get_datasets(label_root: Path) -> list[LabelDataset]:
    """Get dataset and highest resolution scale (i.e. lowest number in zarr multiscale)"""
    data_dirs = sorted(label_root.glob("*/"))
    return [
        LabelDataset(name=dir.name, scale=sorted(dir.glob("*/"))[0].name)
        for dir in data_dirs
    ]


def main() -> None:
    """Evaluate segmentation masks"""

    # Define paths and variables
    gt_root = Path("data/processed/labeled/datasets_2d")
    seg_root = Path("data/processed/segmented/datasets_2d")
    seg_target = "cells"
    models = ["cpsam", "finetuned"]

    datasets = get_datasets(gt_root)
    for dataset, model in tqdm(
        product(datasets, models),
        desc="Evaluating datasets",
        total=len(datasets) * len(models),
    ):
        seg_group_path = seg_root / f"{model}/{dataset.name}.zarr"
        seg_arr_path = f"labels/{seg_target}/masks/{dataset.scale}"

        # Guard against accidentally creating an array
        if not (seg_group_path / seg_arr_path).exists():
            raise FileNotFoundError(
                f"The array at {(seg_group_path / seg_arr_path)} doesn't exist"
            )

        evaluate_segmentation(
            root_path=seg_group_path,
            gt_root_path=gt_root / dataset.name,
            array_path=seg_arr_path,
            seg_target=seg_target,
        )


if __name__ == "__main__":
    main()
