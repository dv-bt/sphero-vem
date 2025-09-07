"""
Register images using pytorch
"""

from dotenv import load_dotenv
import tyro
from tqdm import tqdm
import torch
from tifffile import imwrite
from sphero_vem.registration import register_stack, RegistrationConfig
from sphero_vem.utils import timestamp
from sphero_vem.io import read_tensor


def main():
    load_dotenv(".env")
    config = tyro.cli(RegistrationConfig)
    for shear in [True, False]:
        print(f"Registration with affine transform and {'' if shear else 'no '}shear")
        config.out_dir = (
            config.data_root
            / f"processed/aligned/{config.dataset}/pytorch-run-{timestamp()}"
        )
        config.out_dir.mkdir(parents=True, exist_ok=True)
        config.shear = shear
        register_stack(config)

        # Downscale images
        factor = 5
        ds_dir = config.out_dir / f"downscaled/downscaled-{factor}"
        ds_dir.mkdir(parents=True, exist_ok=True)
        for image_path in tqdm(config.out_dir.glob("*.tif"), "Downscaling images"):
            image = read_tensor(image_path, torch.uint8, 5, False).numpy()
            imwrite(ds_dir / image_path.name, image)


if __name__ == "__main__":
    main()
