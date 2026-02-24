"""
Script to finetune segmentation for cells or nuclei using CellposeSAM on our dataset
"""

from pathlib import Path
from dotenv import load_dotenv
from sphero_vem.segmentation.cellpose import CellposeFinetuneConfig, finetune_cellpose


def main():
    load_dotenv(".env")
    config = CellposeFinetuneConfig(
        dir_labeled=Path("data/processed/labeled/Au_01-vol_01/labeled-06/100-100-100"),
        seg_target="nuclei",
        batch_size=8,
        learning_rate=5e-5,
        n_epochs=301,
    )
    finetune_cellpose(config)


if __name__ == "__main__":
    main()
