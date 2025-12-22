"""
Functions for denoising images, based on Careamics.
"""

from pathlib import Path
from dataclasses import dataclass, field
import zarr
from sklearn.model_selection import train_test_split
import torch
from careamics import CAREamist, Configuration
from careamics.config import create_n2v_configuration
from sphero_vem.utils import timestamp, BaseConfig
from sphero_vem.utils.logging import (
    ArtifactsCallback,
    HyperparamsCallback,
    setup_wanb_env,
)


@dataclass
class DenosingConfig(BaseConfig):
    root_path: Path
    src_path: str
    num_images: int = 10
    val_split: float = 0.2
    random_state: int = 42

    # N2V hyperparameters
    batch_size: int = 128
    patch_size: int = 64
    epochs: int = 100
    unet_depth: int = 2
    unet_num_channels_init: int = 32
    n2v2: bool = False

    # Other params
    num_workers: int = 16
    wandb_project: str = "denoising"
    work_root: Path = Path("data/models/n2v")

    model_name: str = field(init=False)
    work_dir: Path = field(init=False)
    n2v_config: Configuration = field(init=False)

    EXCLUDED_JSON_FIELDS = set(["n2v_config"])
    EXCLUDED_PROCESSING_FIELDS = set(
        [
            "root_path",
            "src_path",
            "num_workers",
            "n2v_config",
            "work_dir",
            "work_root",
            "wandb_project",
        ]
    )

    def __post_init__(self):
        self.model_name = f"n2v-{timestamp()}"
        self.work_dir = self.work_root / self.model_name
        self.n2v_config = create_n2v_configuration(
            experiment_name=self.model_name,
            data_type="array",
            axes="SYX",
            patch_size=[self.patch_size, self.patch_size],
            batch_size=self.batch_size,
            num_epochs=self.epochs,
            use_n2v2=self.n2v2,
            logger="wandb",
            model_params={
                "depth": self.unet_depth,
                "num_channels_init": self.unet_num_channels_init,
            },
            checkpoint_params={"save_top_k": 1},
            train_dataloader_params={"num_workers": self.num_workers},
            val_dataloader_params={"num_workers": self.num_workers},
        )

    def save_n2v_config(self, filepath: str | Path) -> None:
        """Saves the N2V config class to a JSON file."""
        with open(filepath, "w") as file:
            file.write(self.n2v_config.model_dump_json(indent=4))


def train_n2v(config: DenosingConfig) -> None:
    """Train Noise2Void using the parameters specifed in config."""
    root = zarr.open_group(config.root_path, mode="r")
    src_array = root.get(config.src_path)

    # Load training and validation arrays into memory
    train_data, val_data = train_test_split(
        src_array[: config.num_images],
        test_size=config.val_split,
        random_state=config.random_state,
    )

    # Save config files
    config_path = config.work_dir / "config.json"
    n2v_config_path = config.work_dir / "n2v_config.json"

    config.work_dir.mkdir(exist_ok=True, parents=True)
    config.to_json(config_path)
    config.save_n2v_config(n2v_config_path)

    # Set up callbacks
    callback_params = HyperparamsCallback(config.processing_metadata())
    callback_artifacts = ArtifactsCallback([config_path, n2v_config_path])

    # Run training
    setup_wanb_env(config.wandb_project)
    torch.set_float32_matmul_precision("high")
    careamist = CAREamist(
        config.n2v_config,
        work_dir=config.work_dir,
        callbacks=[callback_params, callback_artifacts],
    )
    careamist.train(train_source=train_data, val_source=val_data)
