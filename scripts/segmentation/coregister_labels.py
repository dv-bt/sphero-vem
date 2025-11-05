"""
Bring all segmentation labels in the same space. The reference space is the one of
the NP mask.
"""

from pathlib import Path
import shutil
from tqdm import tqdm
import tifffile
from sphero_vem.postprocessing import upscale_labels
from sphero_vem.io import write_stack


def _copy_manifest(labels_dir: Path, out_dir: Path, seg_target: str) -> None:
    """Copy manifest in the target directory appending seg_target to its name"""
    shutil.copy(labels_dir / "manifest.yaml", out_dir / f"manifest-{seg_target}.yaml")


def _write_np_stack(labels_root: Path, out_dir: Path, dataset: str) -> tuple:
    """Read NPs labels as a stack and write in out_dir. Return stack shape."""
    seg_target = "nps"
    labels_dir = labels_root / seg_target
    out_file = out_dir / f"{dataset}-{seg_target}.tif"
    write_stack(labels_dir, out_file, compressed=True)
    _copy_manifest(labels_dir, out_dir, seg_target)
    with tifffile.TiffFile(out_file) as tif:
        num_pages = len(tif.pages)
        first_page = tif.pages[0]
        target_shape = (num_pages, *first_page.shape)
    return target_shape


def main():
    """Upscale cells and nuclei mask and copy all masks to the same folder.
    Manifests of each mask are copied to the directory with their target appended,
    e.g. manifest-nps.yaml"""
    labels_root = Path("data/processed/segmented/Au_01-vol_01")
    dataset = labels_root.name
    out_dir = labels_root / "coregistered"
    out_dir.mkdir(exist_ok=True)

    # Copy NP mask as a single stack and get shape
    np_mask_shape = _write_np_stack(labels_root, out_dir, dataset)

    targets = ["cells", "nuclei"]
    for seg_target in tqdm(targets, "Upscaling labels"):
        labels_dir = labels_root / f"{seg_target}/merged-labels/"
        labels_name = f"{dataset}-{seg_target}.tif"
        upscale_labels(labels_dir / labels_name, out_dir / labels_name, np_mask_shape)
        _copy_manifest(labels_dir, out_dir, seg_target)


if __name__ == "__main__":
    main()
