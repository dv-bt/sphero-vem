"""
Nanoparticle segmentation
"""

from pathlib import Path
from sphero_vem.utils import timestamp
from sphero_vem.segmentation_np import NanoparticleConfig, NanoparticleSegmentation


def main() -> None:
    """Main script execution"""
    stack_root = Path("data/processed/segmented/Au_01-vol_01.zarr")
    config = NanoparticleConfig(
        stack_root=stack_root, spacing_dir="50-10-10", verbose=True
    )

    # Define model save path
    model_root = Path("data/models/nps")
    model_name = f"nps-{timestamp()}"
    model_dir = model_root / model_name
    model_dir.mkdir(exist_ok=True, parents=True)

    # Fit model
    segmentation = NanoparticleSegmentation(config)
    segmentation.fit()
    segmentation.save(model_dir)
    segmentation.predict(model_name)


if __name__ == "__main__":
    main()
