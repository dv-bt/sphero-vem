"""
Register images using pytorch
"""

from dotenv import load_dotenv
import tyro
from sphero_vem.registration import register_stack, RegistrationConfig
from sphero_vem.utils import timestamp


def main():
    load_dotenv(".env")
    config = tyro.cli(RegistrationConfig)
    config.out_dir = (
        config.data_root
        / f"processed/aligned/{config.dataset}/pytorch-run-{timestamp()}"
    )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    register_stack(config)


if __name__ == "__main__":
    main()
