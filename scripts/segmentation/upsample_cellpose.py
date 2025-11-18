"""
Upsample cellpose masks using distance transform and cell probability
"""

from pathlib import Path
from tqdm import tqdm
from sphero_vem.segmentation import upsample_masks


def main():
    """Main execution"""
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")
    target_spacing = (50, 50, 50)
    for seg_target in tqdm(["cells", "nuclei"]):
        upsample_masks(root_path, seg_target, target_spacing)


if __name__ == "__main__":
    main()
