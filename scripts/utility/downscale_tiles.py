"""
Downscale images by a specified integer factor.
"""

import argparse
from tqdm import tqdm
from pathlib import Path
import torch
from sphero_vem.io import read_tensor, write_image
from sphero_vem.utils import generate_manifest, read_section_errors


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Downscale aligned images")
    parser.add_argument(
        "-f",
        "--factor",
        type=int,
        help="Dowscaling factor",
    )
    parser.add_argument(
        "-s",
        "--source_dir",
        type=Path,
        help="Source directory",
    )
    parser.add_argument(
        "-l",
        "--labels",
        action="store_true",
        help="Flag the tiles as labels and performs nearest neighbor interpolation",
    )
    parser.add_argument(
        "-i", "--isotropic", action="store_true", help="Enforce isotropic voxels"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    factor = args.factor
    extra_fields = {}
    data_dir = args.source_dir
    isotropic_text = "-isotropic" if args.isotropic else ""
    out_dir = data_dir / f"downscaled/downscaled-{factor}{isotropic_text}"
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(data_dir.glob("*.tif"))

    # If factor 10, use every other image to have isotropic voxels (100 nm side)
    if args.isotropic:
        if args.factor == 10:
            section_errors = read_section_errors(data_dir)
            all_images = sorted([image.name for image in images] + section_errors)
            selected = all_images[::2]
            discarded = all_images[1::2]
            images = [
                data_dir / name for name in selected if name not in section_errors
            ]
            extra_fields["discarded"] = discarded
        elif args.factor != 5:
            raise ValueError(
                f"The required downsampling factor {args.factor} is not compatible "
                "with isotropic voxels. Possible values: 5, 10."
            )

    resampling = "nearest" if args.labels else "bilinear"

    for image_path in tqdm(images, desc="Downscaling images"):
        image = read_tensor(
            image_path,
            dtype=torch.uint8,
            ds_factor=factor,
            return_4d=False,
            resample_mode=resampling,
        ).numpy()
        write_image(out_dir / image_path.name, image, compressed=args.labels)

    generate_manifest(
        data_dir.name,
        out_dir,
        images,
        processing=[
            {
                "step": "downscaling",
                "factor": factor,
                "enforce_isotropic": args.isotropic,
            }
        ],
        **extra_fields,
    )


if __name__ == "__main__":
    main()
