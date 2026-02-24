"""
Evaluate prediction accuracy metrics from a segmented stack
"""

from pathlib import Path
from tqdm import tqdm
from sphero_vem.segmentation.cellpose import evaluate_segmentation


def main():
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")
    gt_root_path = Path("data/processed/labeled/Au_01-vol_01/labeled-06")

    label_folders = [
        ("cells", "labels/cells"),
        ("nuclei", "labels/nuclei"),
        ("cells", "pretrained/labels/cells"),
        ("nuclei", "pretrained/labels/cells"),
        ("cells", "amira/cells/filtered"),
        ("nuclei", "amira/nuclei/filtered"),
    ]

    for seg_target, folder in tqdm(label_folders, "Evaluating segmentation"):
        array_path = (
            f"{folder}/masks/50-50-50"
            if "amira" not in folder
            else f"{folder}/50-50-50"
        )
        evaluate_segmentation(
            root_path=root_path,
            gt_root_path=gt_root_path,
            array_path=array_path,
            seg_target=seg_target,
        )


if __name__ == "__main__":
    main()
