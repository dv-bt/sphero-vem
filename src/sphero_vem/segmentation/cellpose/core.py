"""
This module contains functions and classes used for segmentation
"""

import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from cellpose import models, dynamics
import torch
import numpy as np
import zarr
from sphero_vem.io import write_zarr
from sphero_vem.utils import vprint
from sphero_vem.segmentation.cellpose.postptocessing import merge_labels, decompose_flow
from sphero_vem.postprocessing import median_filter


@dataclass
class CellposeFlowConfig:
    """Segment a volume stack using cellpose"""

    root_path: Path
    model: str
    spacing_dir: str
    out_path: Path | None = None
    verbose: bool = True
    zarr_chunks: tuple[int] | None = None
    batch_size: int = 64
    flow3D_smooth: int = 2
    augment: bool = False
    tile_overlap: float = 0.3
    median_filter_cellprob: int | None = 3
    decompose_flows: bool = False
    decompose_flows_pad_fraction: float = 0.3

    seg_target: str = field(init=False)
    model_dir: Path = field(init=False)
    spacing: list[int | float] = field(init=False)
    src_zarr: zarr.Array = field(init=False)

    def __post_init__(self):
        # Allow loading pretrained model
        if self.model == "cpsam":
            # Set model_dir as empty path for compatibility with class init
            self.model_dir = Path("")
            self.seg_target = "cells"
        else:
            self.model_dir = Path(
                f"data/models/cellpose/{self.model}/models/{self.model}"
            )
            self.seg_target = re.search(r"cellposeSAM-(\w+)-", self.model).group(1)

        self.src_zarr = zarr.open_array(
            self.root_path / f"images/{self.spacing_dir}", mode="r"
        )
        self.spacing = self.src_zarr.attrs.get("spacing")

        # If out_dir is not specified, save under the Zarr path of the images
        if not self.out_path:
            self.out_path = self.root_path
        else:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.zarr_chunks:
            self.zarr_chunks = self.src_zarr.chunks


def calculate_flows(config: CellposeFlowConfig) -> None:
    """Segment volume stack using config as input. This function calculates flows and
    cell probability, masks have to be calculated in a following step."""

    processing = [
        {
            "step": "segmentation",
            "model": config.model,
            "seg_target": config.seg_target,
            "batch_size": config.batch_size,
            "flow3D_smooth": config.flow3D_smooth,
            "augment": config.augment,
            "tile_overlap": config.tile_overlap,
            "median_filter_cellprob": config.median_filter_cellprob,
            "decompose_flows": config.decompose_flows,
            "decompose_flows_pad_fraction": config.decompose_flows_pad_fraction,
        }
    ]

    image: np.ndarray = config.src_zarr[:]
    pretrained_model = "cpsam" if config.model == "cpsam" else config.model_dir
    cellpose_model = models.CellposeModel(gpu=True, pretrained_model=pretrained_model)

    time_start = datetime.now()
    vprint(f"Starting segmentation at {time_start}", config.verbose)

    # Shape and channel settings to handle 2D and 3D images correctly
    settings_ndim = {
        2: {"z_axis": None, "channel_axis": None, "do_3D": False},
        3: {
            "z_axis": 0,
            "channel_axis": None,
            "do_3D": True,
        },
    }

    with torch.inference_mode():
        _, flows, _ = cellpose_model.eval(
            image,
            **settings_ndim[image.ndim],
            batch_size=config.batch_size,
            tile_overlap=config.tile_overlap,
            flow3D_smooth=config.flow3D_smooth,
            augment=config.augment,
            compute_masks=False,
        )

    # Ensure GPU memory is garbage collected
    cellpose_model.net.to("cpu")
    del cellpose_model
    torch.cuda.empty_cache()

    time_finish = datetime.now()
    vprint(f"Completed segmentation at {time_finish}", config.verbose)
    vprint(f"Elapsed time: {time_finish - time_start}", config.verbose)

    # Post process flows
    vprint("Starting flow postprocessing", config.verbose)

    dP = np.ascontiguousarray(flows[1]).astype(np.float16, copy=False)
    cellprob = np.ascontiguousarray(flows[2]).astype(np.float16, copy=False)

    if config.decompose_flows:
        dP = decompose_flow(
            dP, config.decompose_flows_pad_fraction, torch.device("cuda")
        )
    if config.median_filter_cellprob:
        cellprob = median_filter(cellprob, config.median_filter_cellprob)

    # Saving flows
    vprint("Saving flows", config.verbose)
    save_root = zarr.open_group(config.out_path, mode="a")

    # Delete old segmentation target group to avoid potential issues withs stale data
    target_group = f"labels/{config.seg_target}"
    if save_root.get(target_group) is not None:
        del save_root[target_group]

    write_zarr(
        save_root,
        cellprob,
        f"{target_group}/flows/cellprob/{config.spacing_dir}",
        src_zarr=config.src_zarr,
        processing=processing,
        zarr_chunks=config.zarr_chunks,
        dtype="f2",
    )

    write_zarr(
        save_root,
        dP,
        f"{target_group}/flows/dP/{config.spacing_dir}",
        src_zarr=config.src_zarr,
        processing=processing,
        zarr_chunks=(3, *config.zarr_chunks),
        dtype="f2",
    )


@dataclass
class CellposeMaskConfig:
    """Parameters used to calculate cellpose masks"""

    root_path: Path
    seg_target: str
    spacing_dir: str = "100-100-100"
    niter: int = 200
    cellprob_threshold: float = -0.5
    flow_threshold: float = 0.4
    min_diam: float = 3
    merge_masks: bool = True
    gaussian_edge_sigma: float = 2.0
    merge_weight_threshold: float = 0.2
    merge_contact_threshold: float = 0.2
    device: str = "cuda"
    zarr_chunks: tuple[int] | None = None

    min_size: int = field(init=False)
    spacing: list[int | float] = field(init=False)

    def __post_init__(self):
        # Celculate min_size in pixel from min_diam in micrometers
        src_zarr = zarr.open_array(
            self.root_path / f"images/{self.spacing_dir}", mode="r"
        )
        self.spacing = src_zarr.attrs.get("spacing")

        # Determine whether min_size should be area or volume
        if len(self.spacing) == 2:
            pixel_um = np.prod(self.spacing) * 1e-6
            min_area_um = np.pi * (self.min_diam / 2) ** 2
            self.min_size = int(min_area_um / pixel_um)
        elif len(self.spacing) == 3:
            voxel_um = np.prod(self.spacing) * 1e-9
            min_vol_um = 4 / 3 * np.pi * (self.min_diam / 2) ** 3
            self.min_size = int(min_vol_um / voxel_um)

        if not self.zarr_chunks:
            self.zarr_chunks = src_zarr.chunks


def calculate_masks(config: CellposeMaskConfig):
    """Calculate segmentation masks using cellpose flows"""

    device = torch.device(config.device)

    root = zarr.open_group(config.root_path, mode="a")
    cellprob_zarr = root.get(
        f"labels/{config.seg_target}/flows/cellprob/{config.spacing_dir}"
    )
    dp_zarr = root.get(f"labels/{config.seg_target}/flows/dP/{config.spacing_dir}")

    cellprob: np.ndarray = cellprob_zarr[:]
    dP: np.ndarray = dp_zarr[:]

    do_3d = True if cellprob.ndim == 3 else False

    masks = dynamics.compute_masks(
        dP=dP,
        cellprob=cellprob,
        niter=config.niter,
        cellprob_threshold=config.cellprob_threshold,
        flow_threshold=config.flow_threshold,
        do_3D=do_3d,
        min_size=config.min_size,
        device=device,
    )

    # Post-process labels
    inputs = [cellprob_zarr.path, dp_zarr.path]
    if config.merge_masks:
        image_arr = root.get(f"images/{config.spacing_dir}")
        inputs.append(image_arr.path)
        image: np.ndarray = image_arr[:]
        masks, _ = merge_labels(
            masks,
            cellprob=cellprob,
            image=image,
            rel_contact_thresh=config.merge_contact_threshold,
            weight_thresh=config.merge_weight_threshold,
            sigma=config.gaussian_edge_sigma,
        )

    processing_dict = asdict(config)
    excluded_keys = ["root_path", "device", "zarr_chunks"]
    processing = cellprob_zarr.attrs.get("processing") + [
        {
            "step": "segmentation mask generation",
            **{
                key: val
                for key, val in processing_dict.items()
                if key not in excluded_keys
            },
        }
    ]

    write_zarr(
        root,
        masks,
        f"labels/{config.seg_target}/masks/{config.spacing_dir}",
        src_zarr=cellprob_zarr,
        dtype=np.uint8 if masks.max() <= 255 else np.uint16,
        inputs=inputs,
        processing=processing,
    )
