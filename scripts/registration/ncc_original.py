"""
Calculate NCC loss (1 - NCC) between images before registration
"""

from pathlib import Path
import json
from tqdm import tqdm
from sphero_vem.io import read_tensor
from sphero_vem.metrics import ncc_loss


def main():
    stack_root = Path(
        "data/processed/denoised/Au_01-vol_01/n2v-depth3-patch128-nimages10"
    )
    out_dir = Path("data/processed/aligned/Au_01-vol_01")

    image_list = sorted(stack_root.glob("*.tif"))
    fixed_list = image_list[:-1]
    moving_list = image_list[1:]
    assert len(fixed_list) == len(moving_list)

    log_data = []

    for i in tqdm(range(len(fixed_list))):
        fixed_path = fixed_list[i]
        moving_path = moving_list[i]

        fixed_image = read_tensor(fixed_path, return_4d=True)
        moving_image = read_tensor(moving_path, return_4d=True)
        ncc = ncc_loss(fixed_image, moving_image)

        log_entry = {
            "pair_index_start": i,
            "fixed_image_path": fixed_path.name,
            "moving_image_path": moving_path.name,
            "initial_loss": float(ncc),
        }

        log_data.append(log_entry)

    # Save summary and transformations
    with open(out_dir / "initial_losses.json", "w") as file:
        json.dump(log_data, file, indent=4)


if __name__ == "__main__":
    main()
