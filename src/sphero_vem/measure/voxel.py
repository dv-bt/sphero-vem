"""Voxel-based shape descriptors and cell assignment."""

import numpy as np
import pandas as pd
from sphero_vem.utils import bbox_expand
from sphero_vem.utils.accelerator import gpu_dispatch, ski, ArrayLike


@gpu_dispatch()
def props_voxel(
    labels: ArrayLike,
    spacing: tuple[float, float, float],
    bbox_margin: int = 15,
    calc_volume: bool = False,
) -> list[dict]:
    """Calculate voxel-based properties using GPU acceleration.

    This function uses skimage.measure.regionprops (or its CuCIM equivalent if CUDA is
    available) calculate per-label properties from a voxel image.

    Parameters
    ----------
    labels : ArrayLike
        Array with the labeled image.
    spacing : tuple[float, float, float]
        Physical spacing of the voxel grid. This will only be used when calc_volume
        is True.
    bbox_margin : int
        Constant margin for bounding box expansion. The returned bounding box will be
        expanded by this value in all directions.

    calc_volume : bool
        Flag that controls whether to calculate volume and related quantities directly
        from the voxel map. This is useful when SDF and mesh-based approaches are
        not feasible.
        Default is False.

    Returns
    -------
    list[dict]
        List of dictionaries containing the calculated properties for each label:
        - label: label ID
        - bbox: bounding box of the object
        - bbox_exp: bounding box expanded by bbox_margin
        - eigvals: eigenvalues of the gyration tensor (sorted ascending)
        - centroid: coordinates of the image centroid
        - volume: label volume in µm^3 (if calc_volume=True)
        - diam_equiv: equivalent diameter in µm^2 (if calc_volume=True)

    """

    props = ski.measure.regionprops(labels)

    results = [
        {
            "label": prop.label,
            "bbox": prop.bbox,
            "bbox_exp": bbox_expand(
                prop.bbox, margin=bbox_margin, im_shape=labels.shape
            ),
            "eigvals": tuple(float(i) for i in sorted(prop.inertia_tensor_eigvals)),
            "centroid": prop.centroid,
        }
        for prop in props
    ]

    if calc_volume:
        props_spacing = ski.measure.regionprops(labels, spacing=spacing)
        for i, prop in enumerate(props_spacing):
            results[i]["volume"] = float(prop.area)
            results[i]["diam_equiv"] = float(prop.equivalent_diameter_area)

    return results


def assign_cell(props: pd.DataFrame, cells: np.ndarray) -> pd.DataFrame:
    """Assign parent cell and return dataframe by looking up centroid

    Parameters
    ----------
    props : pd.DataFrame
        The dataframe containing the region properties. It should have a `"centroid"`
        column containing the tuple indexing the object centroid.
    cells : np.ndarray
        An array containing the cells masks labeled by instance.

    Returns
    -------
    pd.DataFrame
        The updated region properties dataframe with a new column `"parent_cell"`.
    """
    indices = np.array(props["centroid"].tolist()).astype(int)
    props["parent_cell"] = cells[tuple(indices.T)]
    return props
