"""
Script to finetune segmentation for cells or nuclei using CellposeSAM on our dataset
"""

from dotenv import load_dotenv
from sphero_vem.segmentation.cellpose import CellposeFinetuneConfig, finetune_cellpose


def main():
    load_dotenv(".env")
    config = CellposeFinetuneConfig(
        dir_labeled="data/processed/labeled/Au_01-vol_01/labeled-04/50-100-100",
        seg_target="cells",
        batch_size=8,
        save_predictions=False,
        n_epochs=1001,
    )
    finetune_cellpose(config)


if __name__ == "__main__":
    main()
