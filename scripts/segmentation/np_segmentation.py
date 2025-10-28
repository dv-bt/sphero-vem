"""
Nanoparticle segmentation
"""

from pathlib import Path
from tqdm import tqdm
from tifffile import imread
from sphero_vem.io import write_image
from sphero_vem.utils import generate_manifest, timestamp
from sphero_vem.segmentation_np import NanoparticleConfig, NanoparticleSegmentation


if __name__ == "__main__":
    stack_dir = Path("data/processed/cropped/Au_01-vol_01")
    pred_dir = Path("data/processed/segmented/Au_01-vol_01/nps")
    pred_dir.mkdir(exist_ok=True, parents=True)

    model_root = Path("data/models/nps")
    model_name = f"nps-{timestamp()}"
    model_dir = model_root / model_name
    model_dir.mkdir(exist_ok=True, parents=True)

    config = NanoparticleConfig(stack_dir, verbose=True)
    segmentation = NanoparticleSegmentation(config)
    segmentation.fit()
    segmentation.save(model_dir)

    for image_path in tqdm(config.image_list, "Analyzing images"):
        image = imread(image_path)
        posterior_mask, _ = segmentation.predict(image)
        write_image(
            pred_dir / f"{image_path.stem}-nps.tif",
            posterior_mask,
            compressed=True,
        )

    generate_manifest(
        dataset=stack_dir.name,
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
