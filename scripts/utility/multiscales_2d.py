"""
Generate OME-compliant pyramidal image for 2D datasets
"""

from pathlib import Path
import zarr
import skimage as ski
from tqdm import tqdm
import numpy as np
from sphero_vem.io import write_zarr


def get_scales(multiscales: dict) -> list[dict]:
    """Get scales list from multiscales"""

    def _get_scale(ome_dataset: dict) -> list[float]:
        """Get scale for the diven dataset"""
        return ome_dataset["coordinateTransformations"][0]["scale"]

    return [
        {"path": i["path"], "scale": _get_scale(i)} for i in multiscales["datasets"]
    ]


def make_pyramid(root: zarr.Group, factor: int = 2, min_pixel_width: int = 100) -> None:
    """Makae an image pyramid compliant with OME specifications"""
    root = zarr.open_group(root, mode="a")
    group = root.get("images")
    multiscales = group.attrs["multiscales"][0]
    scales = get_scales(multiscales)

    # Read the coarser scale
    scale = scales[-1]
    while (scale["scale"][1] < min_pixel_width) and (
        scale["scale"][0] < min_pixel_width
    ):
        array = group.get(scale["path"])
        image = array[:]
        output_shape = [i // factor for i in image.shape]
        image_ds = ski.transform.resize(
            image, output_shape, order=1, anti_aliasing=True
        )
        if image.dtype == np.uint8:
            image_ds = ski.util.img_as_ubyte(image_ds)
        elif image.dtype == np.uin16:
            image_ds = ski.util.img_as_uint(image_ds)
        new_scale = [i * factor for i in scale["scale"]]
        scales.append({"path": str(int(scale["path"]) + 1), "scale": new_scale})
        scale = scales[-1]
        write_zarr(
            root,
            image_ds,
            dst_path=f"images/{scale['path']}",
            src_zarr=array,
            spacing=new_scale,
            processing={
                "step": "downscaling",
                "algo": "skimage.transform.resize",
                "factor": factor,
                "order": 1,
                "antialiasing": True,
            },
            zarr_chunks=(512, 512),
            multiscales=scales,
        )


def main():
    data_path = Path("data/processed/segmented/datasets_2d")
    factor = 2
    min_pixel_width = 100

    zarr_list = list(data_path.glob("*.zarr"))
    for store in tqdm(zarr_list):
        make_pyramid(store, factor=factor, min_pixel_width=min_pixel_width)


if __name__ == "__main__":
    main()
