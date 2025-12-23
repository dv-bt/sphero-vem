"""Various functions for logging to WandB"""

import logging
from contextlib import contextmanager
import os
from pathlib import Path
from dotenv import load_dotenv
import wandb
from pytorch_lightning.callbacks import Callback


class HyperparamsCallback(Callback):
    """Log hyperparameters to a lightning run."""

    def __init__(self, extra_params: dict):
        self.extra_params = extra_params

    def on_train_start(self, trainer, pl_module):
        if trainer.logger:
            trainer.logger.log_hyperparams(self.extra_params)
            print("Logged extra hyperparameters to WandB.")


class ArtifactsCallback(Callback):
    """Log artifacts to a lightning run."""

    def __init__(self, files: list[Path] | Path):
        super().__init__()
        self.files = files if isinstance(files, list) else [files]

    def on_train_start(self, trainer, pl_module):
        if wandb.run is not None:
            print(f"Uploading artifacts to Run: {wandb.run.name}")
            for file_path in self.files:
                if file_path.exists():
                    wandb.save(file_path, base_path=file_path.parent)
                    print(f"Uploaded: {file_path}")
                else:
                    print(f"Warning: File not found {file_path}")


def setup_wanb_env(project_name: str, experiment_tags: list[str] | str | None = None):
    """Set up a WandB run using environment variables"""

    # Load API keys
    load_dotenv(".env")

    # Set up project and tags
    os.environ["WANDB_PROJECT"] = project_name
    if experiment_tags:
        if isinstance(experiment_tags, list):
            experiment_tags = ",".join(experiment_tags)
        os.environ["WANDB_TAGS"] = experiment_tags
    wandb.login()


@contextmanager
def suppress_logging(loggers: list[str] = None, level: int = logging.ERROR):
    """
    Context manager to temporarily suppress logs below the specified level.

    Restores original levels automatically upon exit.

    Parameters
    ----------
    loggers : list[str] | None
        List of loggers to affect. If None, suppresses pytorch lightning logs.
        Default is None.
    level : int
        Level below which logs are suppressed. Default is loggig.ERROR.
    """
    if loggers is None:
        loggers = ["pytorch_lightning", "lightning.pytorch"]

    original_levels = {}

    try:
        for name in loggers:
            logger = logging.getLogger(name)
            original_levels[name] = logger.level
            logger.setLevel(level)

        yield

    finally:
        for name, old_level in original_levels.items():
            logging.getLogger(name).setLevel(old_level)
