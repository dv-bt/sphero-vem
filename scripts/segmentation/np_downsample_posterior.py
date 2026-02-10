"""
Downsample nanoparticle posterior probability
"""

from pathlib import Path
from sphero_vem.segmentation.np import downsample_posterior


def main():
    """Downsample nanoparticle posterior"""
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")
    sigma = (0.5, 1.5, 1.5)
    dst_spacing = (50, 50, 50)
    src_spacing = (50, 10, 10)

    downsample_posterior(
        root_path=root_path,
        sigma=sigma,
        dst_spacing=dst_spacing,
        src_spacing=src_spacing,
        n_workers=8,
    )


if __name__ == "__main__":
    main()
