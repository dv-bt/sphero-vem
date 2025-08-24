"""
Segment a volume stack using cellpose
"""

import os
from datetime import datetime
from pathlib import Path
import argparse
from dotenv import load_dotenv
import tifffile
from cellpose.models import CellposeModel
from sphero_vem.io import read_stack


load_dotenv(".env")
DATA_ROOT = Path(os.environ.get("DATA_ROOT"))


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Segment volume stack using Cellpose")
    parser.add_argument(
        "-d",
        "--data_dir",
        type=Path,
        help="Source directory",
    )
    args = parser.parse_args()
    args.data_dir = DATA_ROOT / args.data_dir
    return args


def main():
    args = parse_args()

    model_path = (
        DATA_ROOT
        / "models/cellpose/cellposeSAM-cells-ds5-20250823_202550/models/cellposeSAM-cells-ds5-20250823_202550"
    )
    seg_target = "cells"
    dataset = "Au_01-vol_01"
    out_dir = DATA_ROOT / f"segmented/{dataset}/{seg_target}"
    out_dir.mkdir(parents=True, exist_ok=True)

    volume_stack = read_stack(args.data_dir)
    cellpose_model = CellposeModel(gpu=True, pretrained_model=model_path)

    time_start = datetime.now()
    print(f"Starting segmentation at {time_start}")

    masks, _, _ = cellpose_model.eval(
        volume_stack, batch_size=64, do_3D=True, channel_axis=1, z_axis=0
    )

    time_finish = datetime.now()
    print(f"Completed segmentation at {time_finish}")
    print(f"Elapsed time: {time_finish - time_start}")

    tifffile.imwrite(out_dir / f"{dataset}-cells.tif", masks)


if __name__ == "__main__":
    main()
