"""Tests for src/sphero_vem/utils/misc.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sphero_vem.utils.misc import (
    bbox_expand,
    check_isotropic,
    dirname_from_spacing,
    flatten_for_save,
    reconstruct_tuples,
    slice_from_bbox,
    weighted_std,
    temporary_zarr,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tuple_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [1, 2],
            "volume": [1000.0, 2000.0],
            "bbox": [(0, 0, 0, 10, 10, 10), (5, 5, 5, 15, 15, 15)],
            "centroid": [(5.0, 5.0, 5.0), (10.0, 10.0, 10.0)],
        }
    )


# ---------------------------------------------------------------------------
# dirname_from_spacing
# ---------------------------------------------------------------------------


class TestDirnameFromSpacing:
    def test_anisotropic_spacing(self):
        assert dirname_from_spacing((50, 10, 10)) == "50-10-10"

    def test_isotropic_spacing(self):
        assert dirname_from_spacing((100, 100, 100)) == "100-100-100"


# ---------------------------------------------------------------------------
# bbox_expand
# ---------------------------------------------------------------------------


class TestBboxExpand:
    def test_expand_3d(self):
        # Each min decremented by 1, each max incremented by 1
        result = bbox_expand((2, 3, 4, 8, 9, 10), margin=1, im_shape=(20, 20, 20))
        assert result == (1, 2, 3, 9, 10, 11)

    def test_clips_to_bounds(self):
        # Min side clips to 0; max side clips to im_shape dimension
        result = bbox_expand((0, 0, 0, 5, 5, 5), margin=3, im_shape=(6, 6, 6))
        assert result == (0, 0, 0, 6, 6, 6)

    def test_zero_margin_unchanged(self):
        bbox = (2, 3, 7, 8)
        result = bbox_expand(bbox, margin=0, im_shape=(20, 20))
        assert result == bbox

    def test_expand_2d(self):
        result = bbox_expand((5, 5, 15, 15), margin=2, im_shape=(20, 20))
        assert result == (3, 3, 17, 17)


# ---------------------------------------------------------------------------
# slice_from_bbox
# ---------------------------------------------------------------------------


class TestSliceFromBbox:
    def test_3d_slice_indexes_correctly(self):
        bbox = (2, 3, 4, 8, 9, 10)
        slices = slice_from_bbox(bbox)
        assert slices == (slice(2, 8), slice(3, 9), slice(4, 10))
        arr = np.zeros((20, 20, 20))
        assert arr[slices].shape == (6, 6, 6)

    def test_2d_slice(self):
        assert slice_from_bbox((1, 2, 5, 7)) == (slice(1, 5), slice(2, 7))


# ---------------------------------------------------------------------------
# check_isotropic
# ---------------------------------------------------------------------------


class TestCheckIsotropic:
    def test_isotropic_returns_true(self):
        assert check_isotropic((50, 50, 50)) is True

    def test_anisotropic_returns_false(self):
        assert check_isotropic((50, 10, 10), raise_error=False) is False

    def test_anisotropic_raises(self):
        with pytest.raises(ValueError):
            check_isotropic((50, 10, 10), raise_error=True)


# ---------------------------------------------------------------------------
# flatten_for_save / reconstruct_tuples
# ---------------------------------------------------------------------------


class TestFlattenReconstructRoundTrip:
    def test_flatten_creates_indexed_columns(self):
        df = _make_tuple_df()
        flat = flatten_for_save(df)
        # Tuple columns are expanded
        assert "bbox" not in flat.columns
        assert "centroid" not in flat.columns
        for i in range(6):
            assert f"bbox__{i}" in flat.columns
        for i in range(3):
            assert f"centroid__{i}" in flat.columns
        # Scalar columns are preserved
        assert list(flat["label"]) == [1, 2]
        assert list(flat["volume"]) == [1000.0, 2000.0]

    def test_reconstruct_inverts_flatten(self):
        df = _make_tuple_df()
        result = reconstruct_tuples(flatten_for_save(df))
        assert set(result.columns) == set(df.columns)
        # Tuple columns must come back as tuples, not lists
        assert isinstance(result["bbox"].iloc[0], tuple)
        assert isinstance(result["centroid"].iloc[0], tuple)
        assert result["bbox"].iloc[0] == df["bbox"].iloc[0]
        assert result["centroid"].iloc[1] == df["centroid"].iloc[1]

    def test_full_roundtrip_equality(self):
        df = _make_tuple_df()
        result = reconstruct_tuples(flatten_for_save(df))
        pd.testing.assert_frame_equal(result, df, check_like=True)

    def test_custom_sep(self):
        df = _make_tuple_df()
        flat = flatten_for_save(df, sep="||")
        assert "bbox||0" in flat.columns
        result = reconstruct_tuples(flat, sep="||")
        pd.testing.assert_frame_equal(result, df, check_like=True)

    def test_flatten_raises_on_conflicting_column_name(self):
        # A column name that already contains the default sep creates ambiguity
        df = pd.DataFrame({"bbox__0": [(1, 2), (3, 4)]})
        with pytest.raises(ValueError):
            flatten_for_save(df)

    def test_reconstruct_raises_on_noncontiguous_indices(self):
        # Simulate a flattened df where bbox__1
        df = pd.DataFrame(
            {
                "bbox__0": [1, 2],
                "bbox__2": [3, 4],
            }
        )
        with pytest.raises(ValueError, match="Non-contiguous"):
            reconstruct_tuples(df)

    def test_reconstruct_passthrough_nonnumeric_suffix(self):
        # "my__col" contains the separator but suffix is not a digit, must pass through
        df = pd.DataFrame({"my__col": [1, 2], "value": [3.0, 4.0]})
        result = reconstruct_tuples(df)
        assert list(result.columns) == ["my__col", "value"]
        assert list(result["my__col"]) == [1, 2]


# ---------------------------------------------------------------------------
# weighted_std
# ---------------------------------------------------------------------------


class TestWeightedStd:
    def test_uniform_weights_match_numpy_std(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones(5)
        # Both use population std
        assert weighted_std(values, weights) == pytest.approx(np.std(values), abs=1e-10)

    def test_known_weighted_case(self):
        # Weighted mean = (0 * 3 + 10 * 1) / 4 = 2.5
        # Weighted var  = (3 * 6.25 + 56.25) / 4 = 18.75
        # Weighted std  = sqrt(18.75) approx. 4.330
        values = np.array([0.0, 10.0])
        weights = np.array([3.0, 1.0])
        assert weighted_std(values, weights) == pytest.approx(4.330, abs=1e-3)


# ---------------------------------------------------------------------------
# temporary_zarr
# ---------------------------------------------------------------------------


class TestTemporaryZarr:
    def test_temporary_zarr(self):
        with temporary_zarr((4, 8, 8), (1, 8, 8), dtype=np.uint8) as arr:
            # Test array properties
            assert arr.shape == (4, 8, 8)
            assert arr.chunks == (1, 8, 8)
            assert arr.dtype == np.dtype("uint8")

            # Test array is writable
            arr[0] = 1
            assert arr[0, 0, 0] == pytest.approx(1.0, abs=1e-6)
            store_path = arr.store.root
        assert not store_path.exists()

    def test_cleans_up_on_exception(self):
        store_path = None
        with pytest.raises(RuntimeError):
            with temporary_zarr((4, 8, 8), (1, 8, 8)) as arr:
                store_path = arr.store.root
                raise RuntimeError("simulated failure")
        assert store_path is not None and not store_path.exists()

    def test_optional_kwargs(self, tmp_path: Path):
        with temporary_zarr((2, 4, 4), (1, 4, 4), prefix="test_", dir=tmp_path) as arr:
            store_path = arr.store.root
            assert store_path.parent.name.startswith("test_")
            assert store_path.parent.parent == tmp_path
        assert not store_path.exists()
