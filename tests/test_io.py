"""Tests for src/sphero_vem/io.py."""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import tifffile
import yaml
import zarr
from zarr.codecs import BloscCodec

from sphero_vem.io import (
    _create_ome_multiscales,
    _create_zarr_array,
    _get_multiscales,
    _read_manifest,
    _write_zarr_data,
    _write_zarr_metadata,
    repair_multiscales,
    stack_to_zarr,
    write_image,
    write_zarr,
)
from sphero_vem.utils.config import ProcessingStep


# --------------------------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------------------------


def _make_tiff_stack(
    stack_dir: Path,
    n_slices: int,
    shape: tuple[int, int],
    dtype: np.dtype,
) -> np.ndarray:
    """Write n_slices .tif files into stack_dir; return the full (Z,Y,X) array."""
    slices = []
    for i in range(n_slices):
        arr = np.full(shape, fill_value=i + 1, dtype=dtype)
        tifffile.imwrite(stack_dir / f"slice_{i:04d}.tif", arr)
        slices.append(arr)
    return np.stack(slices, axis=0)


def _make_src_zarr(
    root: zarr.Group,
    spacing: tuple,
    processing: list,
) -> zarr.Array:
    """Create a zarr array at 'src/data' with spacing and processing attrs set."""
    group = root.require_group("src")
    arr = group.create_array("data", shape=(4, 8, 8), chunks=(1, 8, 8), dtype=np.uint8)
    arr.attrs["spacing"] = spacing
    arr.attrs["processing"] = processing
    return arr


# --------------------------------------------------------------------------------------
# _read_manifest
# --------------------------------------------------------------------------------------


class TestReadManifest:
    def test_returns_dict_for_valid_manifest(self, tmp_path):
        content = {"processing": [{"step": "denoise"}]}
        (tmp_path / "manifest.yaml").write_text(yaml.dump(content))
        result = _read_manifest(tmp_path)
        assert result == content

    def test_returns_empty_dict_when_no_manifest(self, tmp_path):
        result = _read_manifest(tmp_path)
        assert result == {}


# --------------------------------------------------------------------------------------
# _get_multiscales
# --------------------------------------------------------------------------------------


class TestGetMultiscales:
    def test_sorted_by_ascending_voxel_volume(self, tmp_path):
        root = zarr.open_group(tmp_path / "test.zarr", mode="a")
        coarse = root.create_array(
            "coarse", shape=(4, 8, 8), chunks=(1, 8, 8), dtype=np.uint8
        )
        coarse.attrs["spacing"] = (100, 100, 100)
        fine = root.create_array(
            "fine", shape=(8, 16, 16), chunks=(1, 16, 16), dtype=np.uint8
        )
        fine.attrs["spacing"] = (50, 50, 50)

        result = _get_multiscales(root)

        assert len(result) == 2
        # fine resolution (smaller voxel volume) must come first
        assert list(result[0]["scale"]) == [50, 50, 50]
        assert list(result[1]["scale"]) == [100, 100, 100]

    def test_excludes_array_without_spacing_attr(self, tmp_path):
        root = zarr.open_group(tmp_path / "test.zarr", mode="a")
        with_spacing = root.create_array(
            "with_spacing", shape=(4, 8, 8), chunks=(1, 8, 8), dtype=np.uint8
        )
        with_spacing.attrs["spacing"] = (50, 50, 50)
        root.create_array(
            "no_spacing", shape=(4, 8, 8), chunks=(1, 8, 8), dtype=np.uint8
        )

        result = _get_multiscales(root)

        assert len(result) == 1
        assert result[0]["path"] == "with_spacing"

    def test_returns_empty_list_when_no_spacing_attrs(self, tmp_path):
        root = zarr.open_group(tmp_path / "test.zarr", mode="a")
        root.create_array("arr", shape=(4, 8, 8), chunks=(1, 8, 8), dtype=np.uint8)

        result = _get_multiscales(root)

        assert result == []


# --------------------------------------------------------------------------------------
# _create_ome_multiscales
# --------------------------------------------------------------------------------------


class TestCreateOmeMultiscales:
    def test_3d_spatial_axes_and_scale(self, tmp_path):
        root = zarr.open_group(tmp_path / "test.zarr", mode="a")
        arr = root.create_array(
            "data", shape=(4, 8, 8), chunks=(1, 8, 8), dtype=np.uint8
        )
        arr.attrs["spacing"] = (50, 50, 50)

        _create_ome_multiscales(root)

        multiscales = root.attrs["multiscales"][0]
        axes_names = [a["name"] for a in multiscales["axes"]]
        assert axes_names == ["z", "y", "x"]
        scale = multiscales["datasets"][0]["coordinateTransformations"][0]["scale"]
        assert list(scale) == [50, 50, 50]

    def test_2d_spatial_axes(self, tmp_path):
        root = zarr.open_group(tmp_path / "test.zarr", mode="a")
        arr = root.create_array("data", shape=(8, 8), chunks=(8, 8), dtype=np.uint8)
        arr.attrs["spacing"] = (50, 50)

        _create_ome_multiscales(root)

        multiscales = root.attrs["multiscales"][0]
        axes_names = [a["name"] for a in multiscales["axes"]]
        assert axes_names == ["y", "x"]

    def test_multichannel_prepends_channel_axis(self, tmp_path):
        root = zarr.open_group(tmp_path / "test.zarr", mode="a")
        # 4D array: (C, Z, Y, X) with 3D spacing = multichannel
        arr = root.create_array(
            "data", shape=(2, 4, 16, 16), chunks=(1, 1, 16, 16), dtype=np.uint8
        )
        arr.attrs["spacing"] = (50, 50, 50)

        _create_ome_multiscales(root)

        multiscales = root.attrs["multiscales"][0]
        first_axis = multiscales["axes"][0]
        assert first_axis["name"] == "c"
        assert first_axis["type"] == "channel"
        # channel scale component is 1
        scale = multiscales["datasets"][0]["coordinateTransformations"][0]["scale"]
        assert scale[0] == 1

    def test_path_input_writes_multiscales(self, tmp_path):
        # Tests the `isinstance(group, Path)` branch, which is only reachable via
        # direct call.
        group_path = tmp_path / "g.zarr"
        root = zarr.open_group(group_path, mode="a")
        arr = root.create_array(
            "data", shape=(4, 8, 8), chunks=(1, 8, 8), dtype=np.uint8
        )
        arr.attrs["spacing"] = (50, 50, 50)

        _create_ome_multiscales(group_path)

        root_verify = zarr.open_group(group_path, mode="r")
        assert "multiscales" in root_verify.attrs

    def test_noop_when_no_spacing_arrays(self, tmp_path):
        root = zarr.open_group(tmp_path / "test.zarr", mode="a")
        root.create_array("arr", shape=(4, 8, 8), chunks=(1, 8, 8), dtype=np.uint8)

        _create_ome_multiscales(root)

        assert "multiscales" not in root.attrs


# --------------------------------------------------------------------------------------
# _create_zarr_array
# --------------------------------------------------------------------------------------


class TestCreateZarrArray:
    def test_shape_dtype_chunks_codec(self, tmp_path):
        root = zarr.open_group(tmp_path / "test.zarr", mode="a")
        arr = _create_zarr_array(
            root, dst_path="data", shape=(4, 8, 8), chunks=(1, 8, 8), dtype=np.uint16
        )

        assert arr.shape == (4, 8, 8)
        assert arr.dtype == np.dtype(np.uint16)
        assert arr.chunks == (1, 8, 8)
        assert any(isinstance(c, BloscCodec) for c in arr.metadata.codecs)

    def test_overwrite(self, tmp_path):
        root = zarr.open_group(tmp_path / "test.zarr", mode="a")
        _create_zarr_array(
            root, dst_path="data", shape=(4, 8, 8), chunks=(1, 8, 8), dtype=np.uint8
        )
        # second call with same path must not raise and correctly overwrite the array
        arr = _create_zarr_array(
            root, dst_path="data", shape=(8, 16, 16), chunks=(2, 8, 8), dtype=np.uint16
        )
        assert arr.shape == (8, 16, 16)
        assert arr.dtype == np.dtype(np.uint16)
        assert arr.chunks == (2, 8, 8)


# --------------------------------------------------------------------------------------
# _write_zarr_data
# --------------------------------------------------------------------------------------


class TestWriteZarrData:
    def test_numpy_write(self, tmp_path):
        root = zarr.open_group(tmp_path / "test.zarr", mode="a")
        data = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        arr = _create_zarr_array(
            root, dst_path="data", shape=data.shape, chunks=data.shape, dtype=data.dtype
        )

        _write_zarr_data(dst_zarr=arr, array=data)

        np.testing.assert_array_equal(arr[...], data)

    def test_dask_write(self, tmp_path):
        root = zarr.open_group(tmp_path / "test.zarr", mode="a")
        data = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        arr = _create_zarr_array(
            root, dst_path="data", shape=data.shape, chunks=data.shape, dtype=data.dtype
        )

        _write_zarr_data(dst_zarr=arr, array=da.from_array(data, chunks=data.shape))

        np.testing.assert_array_equal(arr[...], data)

    def test_unsupported_array_type_raises(self, tmp_path):
        root = zarr.open_group(tmp_path / "test.zarr", mode="a")
        arr = _create_zarr_array(
            root, dst_path="data", shape=(2, 3, 4), chunks=(2, 3, 4), dtype=np.uint8
        )

        with pytest.raises(TypeError):
            _write_zarr_data(dst_zarr=arr, array=[[1, 2], [3, 4]])


# --------------------------------------------------------------------------------------
# _write_zarr_metadata
# --------------------------------------------------------------------------------------


class TestWriteZarrMetadata:
    """Tests for _write_zarr_metadata.

    Each test sets up dst_zarr at "images/data" so that the parent group "images" is
    accessible via root.get("images") when the function derives the group path for
    OME multiscales.
    """

    def _setup(self, tmp_path):
        """Return (root, dst_zarr) with dst at 'images/data'."""
        root = zarr.open_group(tmp_path / "test.zarr", mode="a")
        root.require_group("images")
        dst_zarr = _create_zarr_array(
            root,
            dst_path="images/data",
            shape=(4, 8, 8),
            chunks=(1, 8, 8),
            dtype=np.uint8,
        )
        return root, dst_zarr

    def test_explicit_attrs_written(self, tmp_path):
        root, dst_zarr = self._setup(tmp_path)

        _write_zarr_metadata(
            root=root,
            dst_zarr=dst_zarr,
            spacing=(50, 50, 50),
            processing=[],
            inputs=["a", "b"],
        )

        assert tuple(dst_zarr.attrs["spacing"]) == (50, 50, 50)
        assert dst_zarr.attrs["processing"] == []
        assert dst_zarr.attrs["inputs"] == ["a", "b"]

    def test_inherits_spacing_from_src(self, tmp_path):
        root, dst_zarr = self._setup(tmp_path)
        src = _make_src_zarr(root, spacing=(25, 25, 25), processing=[])

        _write_zarr_metadata(root=root, dst_zarr=dst_zarr, src_zarr=src, spacing=None)

        assert tuple(dst_zarr.attrs["spacing"]) == (25, 25, 25)

    def test_prepends_src_processing_history(self, tmp_path):
        root, dst_zarr = self._setup(tmp_path)
        old_step = {"step_name": "ingest", "parameters": {}}
        src = _make_src_zarr(root, spacing=(50, 50, 50), processing=[old_step])
        new_step = ProcessingStep.manual("register", {"lr": 1e-3})

        _write_zarr_metadata(
            root=root,
            dst_zarr=dst_zarr,
            src_zarr=src,
            spacing=(50, 50, 50),
            processing=new_step,
        )

        stored = dst_zarr.attrs["processing"]
        assert stored[0] == old_step
        assert stored[1] == new_step.to_dict()

    @pytest.mark.parametrize(
        "processing",
        [
            ProcessingStep.manual("denoise", {"epochs": 10}),
            {"step_name": "crop", "parameters": {"margin": 5}},
        ],
        ids=["ProcessingStep", "dict"],
    )
    def test_processing_coercion(self, tmp_path, processing):
        # A single ProcessingStep or plain dict (not wrapped in a list) must be
        # stored as [dict]
        root, dst_zarr = self._setup(tmp_path)

        _write_zarr_metadata(
            root=root, dst_zarr=dst_zarr, spacing=(50, 50, 50), processing=processing
        )

        stored = dst_zarr.attrs["processing"]
        assert isinstance(stored, list) and len(stored) == 1

    def test_inputs_defaults_to_src_path(self, tmp_path):
        root, dst_zarr = self._setup(tmp_path)
        src = _make_src_zarr(root, spacing=(50, 50, 50), processing=[])

        _write_zarr_metadata(
            root=root,
            dst_zarr=dst_zarr,
            src_zarr=src,
            spacing=(50, 50, 50),
            inputs=None,
        )

        assert dst_zarr.attrs["inputs"] == [src.path]

    def test_raises_if_no_src_and_no_spacing(self, tmp_path):
        root, dst_zarr = self._setup(tmp_path)
        with pytest.raises(ValueError):
            _write_zarr_metadata(
                root=root, dst_zarr=dst_zarr, src_zarr=None, spacing=None
            )

    def test_raises_if_src_has_no_spacing_attr(self, tmp_path):
        root, dst_zarr = self._setup(tmp_path)
        src_bare = root.require_group("bare").create_array(
            "arr", shape=(4, 8, 8), chunks=(1, 8, 8), dtype=np.uint8
        )
        with pytest.raises(ValueError):
            _write_zarr_metadata(
                root=root, dst_zarr=dst_zarr, src_zarr=src_bare, spacing=None
            )

    def test_multiscales_written_on_parent_group(self, tmp_path):
        root, dst_zarr = self._setup(tmp_path)

        _write_zarr_metadata(root=root, dst_zarr=dst_zarr, spacing=(50, 50, 50))

        assert "multiscales" in root["images"].attrs


# --------------------------------------------------------------------------------------
# write_image  (integration)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
class TestWriteImage:
    def test_uncompressed_roundtrip(self, tmp_path, dtype):
        img = np.arange(64, dtype=dtype).reshape(8, 8)
        path = tmp_path / "img.tif"

        write_image(fname=path, image=img, compressed=False)

        result = tifffile.imread(path)
        np.testing.assert_array_equal(result, img)

    def test_compressed_roundtrip(self, tmp_path, dtype):
        img = np.arange(64, dtype=dtype).reshape(8, 8)
        path = tmp_path / "img.tif"

        write_image(fname=path, image=img, compressed=True)

        result = tifffile.imread(path)
        np.testing.assert_array_equal(result, img)


# --------------------------------------------------------------------------------------
# write_zarr  (integration)
# --------------------------------------------------------------------------------------


class TestWriteZarr:
    """Integration tests for write_zarr.

    Array type dispatch (numpy vs dask) and metadata correctness are covered by
    TestWriteZarrData and TestWriteZarrMetadata respectively. These tests only verify
    that the components are correctly wired together.
    """

    def test_numpy_input_with_zarr_root(self, tmp_path):
        root = zarr.open_group(tmp_path / "test.zarr", mode="a")
        data = np.arange(32, dtype=np.uint8).reshape(2, 4, 4)

        write_zarr(
            root=root,
            array=data,
            dst_path="images/0",
            spacing=(50, 50, 50),
            zarr_chunks=(1, 4, 4),
        )

        dst = root["images"]["0"]
        assert dst.shape == (2, 4, 4)
        assert dst.dtype == np.dtype(np.uint8)
        assert tuple(dst.attrs["spacing"]) == (50, 50, 50)
        assert "multiscales" in root["images"].attrs
        np.testing.assert_array_equal(dst[...], data)

    def test_path_root_accepted(self, tmp_path):
        data = np.zeros((2, 4, 4), dtype=np.uint8)

        write_zarr(
            root=tmp_path / "test.zarr",
            array=data,
            dst_path="images/0",
            spacing=(50, 50, 50),
            zarr_chunks=(1, 4, 4),
        )

        root = zarr.open_group(tmp_path / "test.zarr", mode="r")
        assert root["images"]["0"].shape == (2, 4, 4)

    def test_raises_if_no_chunks_and_no_src(self, tmp_path):
        root = zarr.open_group(tmp_path / "test.zarr", mode="a")
        data = np.zeros((2, 4, 4), dtype=np.uint8)

        with pytest.raises(ValueError):
            write_zarr(
                root=root,
                array=data,
                dst_path="images/0",
                spacing=(50, 50, 50),
                zarr_chunks=None,
                src_zarr=None,
            )

    def test_src_zarr_spacing_and_chunks_inherited(self, tmp_path):
        root = zarr.open_group(tmp_path / "test.zarr", mode="a")
        src = _make_src_zarr(root, spacing=(25, 25, 25), processing=[])

        data = np.zeros((2, 4, 4), dtype=np.uint8)
        write_zarr(
            root=root,
            array=data,
            dst_path="images/0",
            src_zarr=src,
        )

        dst = root["images"]["0"]
        assert tuple(dst.attrs["spacing"]) == (25, 25, 25)
        assert dst.chunks == src.chunks


# --------------------------------------------------------------------------------------
# stack_to_zarr  (integration)
# --------------------------------------------------------------------------------------


class TestStackToZarr:
    def test_converts_stack_to_zarr(self, tmp_path):
        stack_dir = tmp_path / "stack"
        stack_dir.mkdir()
        data = _make_tiff_stack(stack_dir, 3, (16, 16), np.uint8)
        root_path = tmp_path / "out.zarr"

        stack_to_zarr(
            stack_dir=stack_dir,
            root_path=root_path,
            spacing=(50, 10, 10),
            verbose=False,
        )

        root = zarr.open_group(root_path, mode="r")
        arr = root["images"]["50-10-10"]
        assert arr.shape == (3, 16, 16)
        assert tuple(arr.attrs["spacing"]) == (50, 10, 10)
        assert len(arr.attrs["inputs"]) == 3
        assert arr.attrs["processing"] == []
        assert "multiscales" in root["images"].attrs
        np.testing.assert_array_equal(arr[...], data)

    def test_with_manifest(self, tmp_path):
        stack_dir = tmp_path / "stack"
        stack_dir.mkdir()
        _make_tiff_stack(stack_dir, 2, (8, 8), np.uint8)
        processing_entry = [{"step_name": "ingest", "parameters": {}}]
        (stack_dir / "manifest.yaml").write_text(
            yaml.dump({"processing": processing_entry})
        )
        root_path = tmp_path / "out.zarr"

        stack_to_zarr(
            stack_dir=stack_dir,
            root_path=root_path,
            spacing=(50, 10, 10),
            verbose=False,
        )

        root = zarr.open_group(root_path, mode="r")
        arr = root["images"]["50-10-10"]
        assert arr.attrs["processing"] == processing_entry

    def test_raises_for_none_spacing(self, tmp_path):
        stack_dir = tmp_path / "stack"
        stack_dir.mkdir()
        _make_tiff_stack(stack_dir, 2, (8, 8), np.uint8)

        with pytest.raises(NotImplementedError):
            stack_to_zarr(
                stack_dir=stack_dir, root_path=tmp_path / "out.zarr", spacing=None
            )


# --------------------------------------------------------------------------------------
# repair_multiscales  (integration)
# --------------------------------------------------------------------------------------


class TestRepairMultiscales:
    def test_repairs_stale_multiscales_in_flat_group(self, tmp_path):
        store_path = tmp_path / "test.zarr"
        root = zarr.open_group(store_path, mode="a")
        images_group = root.require_group("images")
        arr1 = _create_zarr_array(
            root, dst_path="images/0", shape=(4, 8, 8), chunks=(1, 8, 8), dtype=np.uint8
        )
        arr1.attrs["spacing"] = (50, 50, 50)
        arr2 = _create_zarr_array(
            root, dst_path="images/1", shape=(2, 4, 4), chunks=(1, 4, 4), dtype=np.uint8
        )
        arr2.attrs["spacing"] = (100, 100, 100)
        # Seed a stale/empty multiscales entry to trigger repair
        images_group.attrs["multiscales"] = {}

        repair_multiscales(root=store_path)

        root_verify = zarr.open_group(store_path, mode="r")
        multiscales = root_verify["images"].attrs["multiscales"]
        assert isinstance(multiscales, list)
        assert len(multiscales[0]["datasets"]) == 2

    def test_recursive_repair(self, tmp_path):
        store_path = tmp_path / "test.zarr"
        root = zarr.open_group(store_path, mode="a")
        inner = root.require_group("a/b")
        arr = _create_zarr_array(
            root, dst_path="a/b/data", shape=(4, 8, 8), chunks=(1, 8, 8), dtype=np.uint8
        )
        arr.attrs["spacing"] = (50, 50, 50)
        # Seed multiscales on the inner group to trigger repair when recursion reaches it
        inner.attrs["multiscales"] = {}

        repair_multiscales(root=store_path)

        root_verify = zarr.open_group(store_path, mode="r")
        assert "multiscales" in root_verify["a"]["b"].attrs
        assert isinstance(root_verify["a"]["b"].attrs["multiscales"], list)
        # Root itself was never seeded with "multiscales"
        assert "multiscales" not in root_verify.attrs

    def test_start_path_limits_scope(self, tmp_path):
        store_path = tmp_path / "test.zarr"
        root = zarr.open_group(store_path, mode="a")
        sub = root.require_group("sub")
        arr = _create_zarr_array(
            root, dst_path="sub/data", shape=(4, 8, 8), chunks=(1, 8, 8), dtype=np.uint8
        )
        arr.attrs["spacing"] = (50, 50, 50)
        sub.attrs["multiscales"] = {}  # seed only on sub

        repair_multiscales(root=store_path, start_path="sub")

        root_verify = zarr.open_group(store_path, mode="r")
        assert isinstance(root_verify["sub"].attrs["multiscales"], list)
        assert "multiscales" not in root_verify.attrs

    def test_noop_for_group_without_multiscales_attr(self, tmp_path):
        # repair_multiscales only repairs groups that already have a "multiscales"
        # key; groups missing the key entirely are left untouched.
        store_path = tmp_path / "test.zarr"
        root = zarr.open_group(store_path, mode="a")
        arr = _create_zarr_array(
            root, dst_path="data", shape=(4, 8, 8), chunks=(1, 8, 8), dtype=np.uint8
        )
        arr.attrs["spacing"] = (50, 50, 50)
        # Deliberately do NOT seed "multiscales" on root

        repair_multiscales(root=store_path)

        root_verify = zarr.open_group(store_path, mode="r")
        assert "multiscales" not in root_verify.attrs
