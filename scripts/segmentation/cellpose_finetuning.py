"""
Script to finetune segmentation for cells or nuclei using CellposeSAM on our dataset
"""

from dotenv import load_dotenv
import argparse
from sphero_vem.segmentation import CellposeConfig, finetune_cellpose

load_dotenv(".env")


def parse_arguments() -> CellposeConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, help="Dataset path")
    parser.add_argument(
        "--ds", type=int, default=CellposeConfig.downscaling, help="Dowscaling"
    )
    parser.add_argument(
        "--lr", type=float, default=CellposeConfig.learning_rate, help="Learning rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=CellposeConfig.batch_size, help="Batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=CellposeConfig.n_epochs, help="Number of epochs"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=CellposeConfig.test_size,
        help="Test dataset fraction",
    )
    parser.add_argument(
        "--seg-target",
        type=str,
        default=CellposeConfig.seg_target,
        help="Segmentation target: 'cells' or 'nuclei'",
    )
    parser.add_argument(
        "--use-float32",
        action="store_true",
        help="Use float32 for model weights. Otherwise use bfloat16 (default)",
    )

    args = parser.parse_args()

    return CellposeConfig(
        dir_labeled=args.dataset,
        downscaling=args.ds,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        test_size=args.test_size,
        seg_target=args.seg_target,
    )


def main():
    config = parse_arguments()
    finetune_cellpose(config)


if __name__ == "__main__":
    main()
