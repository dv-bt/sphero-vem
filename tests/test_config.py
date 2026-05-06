"""Tests for src/sphero_vem/utils/config.py."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import pytest

import numpy as np
from sphero_vem.utils.config import BaseConfig, ProcessingStep, to_serializable


# ---------------------------------------------------------------------------
# BaseConfig subclass used throughout the module
# ---------------------------------------------------------------------------


@dataclass
class _SampleConfig(BaseConfig):
    path_field: Path = Path("/tmp/data.zarr")
    spacing: tuple[float, float, float] = (50.0, 10.0, 10.0)
    count: int = 7
    aux: float = 3.14
    secret: str = ""

    EXCLUDED_JSON_FIELDS: ClassVar[set[str]] = frozenset({"secret"})
    EXCLUDED_PROCESSING_FIELDS: ClassVar[set[str]] = frozenset({"aux"})


# ---------------------------------------------------------------------------
# to_serializable
# ---------------------------------------------------------------------------


class TestToSerializable:
    def test_primitive_coercions(self):
        result = to_serializable({"p": Path("/a/b/c"), "t": (1, 2, 3)})
        assert isinstance(result["p"], str)
        assert result["p"] == "/a/b/c"
        assert isinstance(result["t"], list)
        assert result["t"] == [1, 2, 3]

    def test_nested_dict(self):
        result = to_serializable({"outer": {"path": Path("/x"), "tup": (4.0, 5.0)}})
        assert isinstance(result["outer"]["path"], str)
        assert isinstance(result["outer"]["tup"], list)

    def test_numpy_coercions(self):
        result = to_serializable(
            {
                "f": np.float32(3.14),
                "i": np.int16(123),
                "a": np.arange(6, dtype=np.int16),
            }
        )
        assert isinstance(result["f"], float)
        assert isinstance(result["i"], int)
        assert isinstance(result["a"], list)
        assert all(isinstance(i, int) for i in result["a"])

    def test_all_values_json_native(self):
        result = to_serializable(
            {
                "a": Path("/x"),
                "b": (1, 2),
                "c": 42,
                "d": "hello",
                "e": 3.14,
                "f": True,
                "g": None,
            }
        )
        # Round-trip through json must not raise
        json.dumps(result)


# ---------------------------------------------------------------------------
# BaseConfig
# ---------------------------------------------------------------------------


class TestBaseConfig:
    def test_full_config_serialization(self):
        cfg = _SampleConfig()
        fc = cfg.full_config()
        assert isinstance(fc, dict)
        assert "secret" not in fc
        assert "aux" in fc
        assert "spacing" in fc
        assert "count" in fc

    def test_processing_metadata_serialization(self):
        cfg = _SampleConfig()
        pm = cfg.processing_metadata()
        assert isinstance(pm, dict)
        assert "secret" not in pm
        assert "aux" not in pm
        assert "spacing" in pm
        assert "count" in pm

    def test_roundtrip(self, tmp_path):
        cfg = _SampleConfig()
        p = tmp_path / "cfg.json"
        cfg.to_json(p)
        restored = _SampleConfig.from_json(p)

        assert restored == cfg

    def test_parameter_coercion(self):
        path_str = "/tmp/data.zarr"
        spacing_list = [50.0, 10.0, 10.0]
        count_np = np.int16(7)
        aux_np = np.float32(3.14)

        cfg = _SampleConfig(
            path_field=path_str, spacing=spacing_list, count=count_np, aux=aux_np
        )

        assert cfg.path_field == Path(path_str)
        assert cfg.spacing == (50.0, 10.0, 10.0)
        assert isinstance(cfg.count, int)
        assert cfg.count == int(count_np)
        assert isinstance(cfg.aux, float)
        assert cfg.aux == pytest.approx(float(aux_np))


# ---------------------------------------------------------------------------
# ProcessingStep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", [None, "1.0.0"])
class TestProcessingStep:
    def test_init_from_config(self, version):
        cfg = _SampleConfig()
        step = ProcessingStep.from_config("denoise", cfg, version=version)
        assert step.step_name == "denoise"
        assert isinstance(step.timestamp, str)
        # raises if not a valid ISO format string
        datetime.fromisoformat(step.timestamp)
        assert step.parameters == cfg.processing_metadata()
        assert step.version == version

    def test_init_manual(self, version):
        params = {"margin": 10, "axis": "z"}
        step = ProcessingStep.manual("crop", params, version=version)
        assert step.step_name == "crop"
        assert isinstance(step.timestamp, str)
        # raises if not a valid ISO format string
        datetime.fromisoformat(step.timestamp)
        assert step.parameters == params
        assert step.version == version

    def test_dict_roundtrip(self, version):
        original = ProcessingStep.manual("register", {"lr": 1e-3}, version=version)
        restored = ProcessingStep.from_dict(original.to_dict())
        assert restored.step_name == original.step_name
        assert restored.timestamp == original.timestamp
        assert restored.parameters == original.parameters
        assert restored.version == original.version
