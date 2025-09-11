"""
Segment a volume stack using cellpose
"""

import os
import re
from datetime import datetime
from pathlib import Path
import argparse
from dotenv import load_dotenv
from cellpose.models import CellposeModel
from sphero_vem.io import read_stack, imwrite
from sphero_vem.utils import generate_manifest, timestamp


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
    parser.add_argument("-m", "--model", type=str, help="Model name")
    args = parser.parse_args()
    args.data_dir = DATA_ROOT / args.data_dir
    return args


def match_model_target(model_name: str) -> str:
    match = re.search(r"cellposeSAM-(\w+)-", model_name)
    return match.group(1)


def main():
    args = parse_args()

    model_path = DATA_ROOT / f"models/cellpose/{args.model}/models/{args.model}"

    # Parameters
    dataset = "Au_01-vol_01"

    seg_params = {
        "step": "segmentation",
        "model": args.model,
        "seg_target": match_model_target(args.model),
        "batch_size": 128,
    }

    out_dir = (
        DATA_ROOT
        / f"processed/segmented/{dataset}/{seg_params['seg_target']}-run{timestamp()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_manifest(
        dataset, out_dir, sorted(args.data_dir.glob("*.tif")), [seg_params]
    )

    volume_stack = read_stack(args.data_dir)
    cellpose_model = CellposeModel(gpu=True, pretrained_model=model_path)

    time_start = datetime.now()
    print(f"Starting segmentation at {time_start}")

    masks, _, _ = cellpose_model.eval(
        volume_stack,
        batch_size=seg_params["batch_size"],
        do_3D=True,
        channel_axis=1,
        z_axis=0,
    )

    time_finish = datetime.now()
    print(f"Completed segmentation at {time_finish}")
    print(f"Elapsed time: {time_finish - time_start}")

    imwrite(
        out_dir / f"{dataset}-{seg_params['seg_target']}.tif", masks, uncompressed=False
    )


if __name__ == "__main__":
    main()
