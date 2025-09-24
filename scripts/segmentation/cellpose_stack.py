"""
Segment a volume stack using cellpose
"""

import re
from dataclasses import dataclass, field, asdict
import tyro
from datetime import datetime
from pathlib import Path
from cellpose.models import CellposeModel
from sphero_vem.io import read_stack, imwrite
from sphero_vem.utils import generate_manifest, timestamp


@dataclass
class SegmentationParams:
    """Parameters passed to the segmentation function"""

    batch_size: int = 64
    flow3D_smooth: int = 0
    stitch_threshold: float = 0.0
    cellprob_threshold: float = 0.0


@dataclass
class Config:
    """Segment a volume stack using cellpose"""

    data_dir: Path
    model: str
    seg_params: SegmentationParams

    dataset: str = field(init=False)
    seg_target: str = field(init=False)
    model_dir: Path = field(init=False)
    out_dir: Path = field(init=False)
    out_path: Path = field(init=False)

    def __post_init__(self):
        self.dataset = re.search(r"(Au_\d+-vol_\d+)", str(self.data_dir)).group(1)
        self.seg_target = re.search(r"cellposeSAM-(\w+)-", self.model).group(1)
        self.model_dir = Path(f"data/models/cellpose/{self.model}/models/{self.model}")
        self.out_dir = Path(
            f"data/processed/segmented/{self.dataset}/{self.seg_target}-run{timestamp()}"
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_path = self.out_dir / f"{self.dataset}-{self.seg_target}.tif"


def main():
    config = tyro.cli(Config)
    seg_params = asdict(config.seg_params)

    processing = [
        {
            "step": "segmentation",
            "model": config.model,
            "seg_target": config.seg_target,
            **seg_params,
        }
    ]

    generate_manifest(
        config.dataset,
        config.out_dir,
        sorted(config.data_dir.glob("*.tif")),
        processing,
    )

    volume_stack = read_stack(config.data_dir)
    cellpose_model = CellposeModel(gpu=True, pretrained_model=config.model_dir)

    time_start = datetime.now()
    print(f"Starting segmentation at {time_start}")

    masks, _, _ = cellpose_model.eval(
        volume_stack, do_3D=True, channel_axis=1, z_axis=0, **seg_params
    )

    time_finish = datetime.now()
    print(f"Completed segmentation at {time_finish}")
    print(f"Elapsed time: {time_finish - time_start}")

    imwrite(config.out_path, masks, uncompressed=False)


if __name__ == "__main__":
    main()
