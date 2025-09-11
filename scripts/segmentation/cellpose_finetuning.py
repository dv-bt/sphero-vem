"""
Script to finetune segmentation for cells or nuclei using CellposeSAM on our dataset
"""

from dotenv import load_dotenv
import tyro
from sphero_vem.segmentation import CellposeConfig, finetune_cellpose


def main():
    load_dotenv(".env")
    config = tyro.cli(CellposeConfig)
    finetune_cellpose(config)


if __name__ == "__main__":
    main()
