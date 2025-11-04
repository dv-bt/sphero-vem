"""
Nanoparticle segmentation
"""

from pathlib import Path
from tqdm import tqdm
import tyro
from tifffile import imread
from sphero_vem.io import write_image
from sphero_vem.utils import generate_manifest, timestamp, infer_dataset
from sphero_vem.segmentation_np import NanoparticleConfig, NanoparticleSegmentation


def main() -> None:
    """Main script execution"""
    config = tyro.cli(NanoparticleConfig)

    # Define prediction save paths
    dataset = infer_dataset(config.stack_dir)
    pred_dir = Path(f"data/processed/segmented/{dataset}/nps")
    pred_dir.mkdir(exist_ok=True, parents=True)

    # Define model save path
    model_root = Path("data/models/nps")
    model_name = f"nps-{timestamp()}"
    model_dir = model_root / model_name
    model_dir.mkdir(exist_ok=True, parents=True)

    # Fit model
    segmentation = NanoparticleSegmentation(config)
    segmentation.fit()
    segmentation.save(model_dir)

    # Predict NPs for images in the stack
    for image_path in tqdm(config.image_list, "Analyzing images"):
        image = imread(image_path)
        posterior_mask, _ = segmentation.predict(image)
        write_image(
            pred_dir / f"{image_path.stem}-nps.tif",
            posterior_mask,
            compressed=True,
        )

    generate_manifest(
        dataset=dataset,
        out_dir=pred_dir,
        images=config.image_list,
        processing=[
            {
                "step": "segmentation",
                "seg_target": "nps",
                "model": model_name,
                **config.to_serializable(),
            }
        ],
    )


if __name__ == "__main__":
    main()
