"""Tests for src/sphero_vem/metrics.py."""

from __future__ import annotations

import pytest
import torch

from sphero_vem.metrics import LossDispatcher, ncc_loss


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

# Spatial dims
_H, _W = 32, 32


def _rand_pair(dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    img1 = torch.rand(1, 1, _H, _W, dtype=dtype)
    img2 = torch.rand(1, 1, _H, _W, dtype=dtype)
    return img1, img2


# ---------------------------------------------------------------------------
# ncc_loss
# ---------------------------------------------------------------------------


class TestNccLoss:
    def test_identical_tensors_zero_loss(self):
        # Perfect positive correlation: 1 − 1 = 0
        x, _ = _rand_pair()
        assert ncc_loss(x, x).item() == pytest.approx(0.0, abs=1e-4)

    def test_negated_tensors_max_loss(self):
        # Perfect anti-correlation: 1 − (−1) = 2
        x, _ = _rand_pair()
        assert ncc_loss(x, -x).item() == pytest.approx(2.0, abs=1e-4)

    def test_constant_tensor_returns_finite(self):
        # Zero-variance input must not produce NaN or inf
        x = torch.ones(1, 1, _H, _W)
        loss = ncc_loss(x, x)
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype(self, dtype: torch.dtype):
        x, _ = _rand_pair(dtype=dtype)
        loss = ncc_loss(x, x)
        assert torch.isfinite(loss)
        assert 0.0 <= loss.item() <= 2.0


# ---------------------------------------------------------------------------
# LossDispatcher
# ---------------------------------------------------------------------------


class TestLossDispatcher:
    @pytest.mark.parametrize("name", ["mse", "mae", "ncc", "ssim"])
    def test_valid_names_instantiate(self, name: str):
        # must not raise
        LossDispatcher(name)

    def test_invalid_name_raises(self):
        # error message should list valid options, mse is proxy
        with pytest.raises(ValueError, match="mse"):
            LossDispatcher("bad_loss")

    @pytest.mark.parametrize("name", ["mse", "mae", "ncc"])
    def test_mse_mae_ncc_return_scalar(self, name: str):
        img1, img2 = _rand_pair()
        result = LossDispatcher(name)(img1, img2)
        assert result.ndim == 0
        assert torch.isfinite(result)

    def test_ssim_returns_scalar(self):
        # kornia.losses.ssim_loss requires window_size as its third argument
        img1, img2 = _rand_pair()
        result = LossDispatcher("ssim")(img1, img2, window_size=5)
        assert result.ndim == 0
        assert torch.isfinite(result)
