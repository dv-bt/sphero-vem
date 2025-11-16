"""
Nanoparticle segmentation
"""

import json
from typing import Self
from pathlib import Path
from dataclasses import dataclass, asdict
from tqdm import tqdm
import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle
from scipy.optimize import minimize_scalar
from scipy.ndimage import find_objects
from sphero_vem.utils import CustomJSONEncoder, create_ome_multiscales
from sphero_vem.utils.accelerator import (
    xp,
    gpu_dispatch,
    ArrayLike,
    ndi,
    to_host,
    to_device,
)


@dataclass
class NanoparticleConfig:
    """Configuration class for NP segmentation"""

    stack_root: Path
    spacing_dir: "str"
    verbose: bool = False
    max_iter: int = 10000
    eps: float = 1e-12
    nll_tol: float = 1e-10
    sampling_step: int = 1
    percent_th_low: float = 99.5
    percent_th_high: float = 99.8
    halo_pad: int = 20
    min_size: int = 20
    posterior_th: float = 0.95
    beta_params: tuple[float, float] = (1.0, 20.0)
    save_prob: bool = False

    def __post_init__(self) -> None:
        """Get image list"""

        # Enforce correct types
        if isinstance(self.beta_params, list):
            self.beta_params = tuple(self.beta_params)
        if isinstance(self.stack_root, str):
            self.stack_root = Path(self.stack_root)

    def to_serializable(self) -> dict:
        """Return the dataclass as a JSON or YAML-serializable dictionary"""
        config_dict = asdict(self)
        json_string = json.dumps(config_dict, cls=CustomJSONEncoder)
        return json.loads(json_string)

    def save_json(self, filepath: str | Path) -> None:
        """Saves the dataclass instance to a JSON file."""
        with open(filepath, "w") as file:
            json.dump(self.to_serializable(), file, indent=4)

    @classmethod
    def from_json(cls, filepath: str | Path) -> Self:
        """Loads a dataclass instance from a JSON file."""

        with open(filepath, "r") as file:
            config_dict = json.load(file)
        return cls(**config_dict)


class NanoparticleSegmentation:
    """Calculate posterior probability of nanoparticles using the EM algorithm"""

    def __init__(
        self,
        config: NanoparticleConfig,
    ) -> None:
        """Initialize the class from patches of NPs and background.

        Parameters
        ----------
        dir : Path
            Directory containing the patches. The directory is expected to contain two
            subdirectories named "np-crops" and "bg-crops"
        verbose : bool
            Enable verbose output. Default is False.
        """
        self.config = config
        self.stack_root = zarr.open_group(self.config.stack_root, mode="a")
        self.volume_stack: zarr.Array = self.stack_root["images"][
            self.config.spacing_dir
        ]

    @classmethod
    def load(cls, model_dir: str | Path) -> Self:
        """Load a pretrained model from a directory. The directory is expected to have
        a training_manifest.json and model_params.npz of the correct format"""
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        config = NanoparticleConfig.from_json(model_dir / "training_manifest.json")
        segmentation = cls(config)

        params = np.load(model_dir / "model_params.npz")
        segmentation.p_np = params["p_np"]
        segmentation.p_bg = params["p_bg"]
        segmentation.p_stack = params["p_stack"]
        segmentation.hist_stack = params["hist_stack"]
        segmentation.hist_bg = params["hist_bg"]

        return segmentation

    def save(self, target_dir: str | Path) -> None:
        """Save the calculated probabilities and the training parameters in the
        specified directory as model_params.npz and training_manifest.json, respectively.
        It also saves the summary of the fit results in fit_results.json"""
        if isinstance(target_dir, str):
            target_dir = Path(target_dir)
        self.config.save_json(target_dir / "training_manifest.json")
        np.savez(
            target_dir / "model_params.npz",
            p_np=self.p_np,
            p_bg=self.p_bg,
            p_stack=self.p_stack,
            hist_stack=self.hist_stack,
            hist_bg=self.hist_bg,
        )
        with open(target_dir / "fit_results.json", "w") as file:
            json.dump(self.summary_fit, file, indent=4, cls=CustomJSONEncoder)

    def _normalize_pmf(self, hist: np.ndarray) -> np.ndarray:
        """Normalizes a count histogram and adds a tiny floor to avoid divisions by zero"""
        pmf = hist.astype(np.float64)
        if pmf.sum() == 0:
            return np.full(256, 1 / 256, dtype=np.float64)
        pmf /= pmf.sum()
        # Add floor and re-normalize
        pmf = np.maximum(pmf, self.config.eps)
        pmf /= pmf.sum()
        return pmf

    def _calc_stack_hist(self) -> None:
        """Calculate the histogram of the entire stack with the selected sampling step."""
        hist_stack = np.zeros(256, dtype=np.int64)
        for idx in tqdm(
            range(0, self.volume_stack.shape[0], self.config.sampling_step),
            "Calculating stack histogram",
            disable=not self.config.verbose,
        ):
            image = self.volume_stack[idx]
            hist_stack += bincount_ubyte(image)
        self.hist_stack = hist_stack
        self.p_stack = self._normalize_pmf(hist_stack)

    def _calc_bg_hist(self) -> None:
        """Calculate background histogram."""

        hist_bg = np.zeros(256, dtype=np.int64)
        self.th_low = np.searchsorted(
            self.p_stack.cumsum(), self.config.percent_th_low / 100
        )
        self.th_high = np.searchsorted(
            self.p_stack.cumsum(), self.config.percent_th_high / 100
        )

        for idx in tqdm(
            range(0, self.volume_stack.shape[0], self.config.sampling_step),
            desc="Calculating background histogram",
            disable=not self.config.verbose,
        ):
            image = self.volume_stack[idx]
            hist_bg += self._extract_bg_hist(image)

        self.hist_bg = hist_bg
        self.p_bg = self._normalize_pmf(hist_bg)

    @gpu_dispatch(return_to_host=True)
    def _extract_bg_hist(self, image: ArrayLike) -> ArrayLike | None:
        """Calculate background histogram with GPU acceleration, if available.
        If no background found, return 0. This is to keep consistency with background
        accumulation across slices."""

        def find_objects_cpu(labels: ArrayLike) -> list[tuple[slice]]:
            """scipy.ndimage.find_objects is not yet implemented in cupy.
            Fallback to CPU and handle moving inputs"""
            labels = to_host(labels)
            return find_objects(labels)

        # Rough object seeds by intensity
        mask = image > self.th_low
        labels, num = ndi.label(mask, structure=np.ones((3, 3), dtype=xp.uint8))
        if num > 0:
            sizes = xp.bincount(labels.ravel())[1:]
            keep = xp.nonzero(sizes >= self.config.min_size)[0]
            bboxes = find_objects_cpu(labels)
            obj_mask = xp.zeros(image.shape, dtype=bool)
            for i in keep:
                sl = self._expand_bbox(bboxes[int(i)], image.shape)
                obj_mask[sl] = True
        else:
            obj_mask = xp.zeros(image.shape, dtype=bool)

        # Hard background exclusion above upper intensity threshold
        bright_mask = image >= self.th_high
        bg_mask = (~obj_mask) & (~bright_mask)
        if bg_mask.any():
            return bincount_ubyte(image[bg_mask])
        return

    def _expand_bbox(self, bbox: tuple[slice], image_shape: tuple) -> tuple[slice]:
        """Expand a ndimage.find_objects bbox by pad pixels controlled by config.halo_pad."""
        (slice_y, slice_x) = bbox
        y0 = max(slice_y.start - self.config.halo_pad, 0)
        y1 = min(slice_y.stop + self.config.halo_pad, image_shape[0])
        x0 = max(slice_x.start - self.config.halo_pad, 0)
        x1 = min(slice_x.stop + self.config.halo_pad, image_shape[1])
        return (slice(y0, y1), slice(x0, x1))

    def _init_w_bg(self) -> np.float64:
        """Initialize w_bg as ratio of background/total pixels"""
        return self.hist_bg.sum() / self.hist_stack.sum()

    def _init_p_np(self) -> np.ndarray:
        """Initialize NP probability to match tail of stack distribution"""
        p_np = np.zeros_like(self.p_stack, dtype=np.float64)
        p_np[self.th_low :] = self.p_stack[self.th_low :]
        return self._normalize_pmf(p_np)

    def _deconvolve_mixture(self):
        """Plain EM with tail init for p_obj and 1-D NLL solve for w each M-step."""

        # Initialization
        w_bg = self._init_w_bg()
        p_np = self._init_p_np()

        def neg_log_likelihood(w_bg: float) -> np.float64:
            """Calculate complete negative data log likelihood"""
            mix = w_bg * self.p_bg + (1 - w_bg) * p_np
            return -np.sum(self.p_stack * np.log(mix + self.config.eps))

        nll_prev = np.inf
        for i in range(self.config.max_iter):
            # E-step
            log_bg = np.log(w_bg + self.config.eps) + np.log(
                self.p_bg + self.config.eps
            )
            log_np = np.log(1 - w_bg + self.config.eps) + np.log(p_np + self.config.eps)
            t = np.maximum(log_bg, log_np)
            den = t + np.log(np.exp(log_bg - t) + np.exp(log_np - t))
            gamma_bg = np.exp(log_bg - den)

            # M-step
            q = (1.0 - gamma_bg) * self.p_stack
            if q.sum() > 0:
                p_np = self._normalize_pmf(q)
            else:
                # keep previous p_np
                p_np = p_np

            # M-step for w_bg: minimize NLL
            result_scalar = minimize_scalar(
                neg_log_likelihood,
                bounds=(self.config.eps, 1 - self.config.eps),
                method="bounded",
            )
            w_bg = result_scalar.x

            nll = neg_log_likelihood(w_bg)
            if -(nll - nll_prev) < self.config.nll_tol:
                break
            nll_prev = nll

        self.p_np = p_np
        self.summary_fit = {"w_bg": w_bg, "nll": nll, "nit": i}

    def fit(self) -> None:
        """Find the NP probability density"""
        self._calc_stack_hist()
        self._calc_bg_hist()
        self._deconvolve_mixture()

    def _fit_pi(self, hist_image):
        """Find mixing weight pi by minimizing negative data log likelihood with beta
        prior."""

        def nll_beta_prior(pi):
            mix = pi * self.p_np + (1 - pi) * self.p_bg
            ll = np.sum(hist_image * np.log(mix + self.config.eps))
            beta_prior = (
                (self.config.beta_params[0] - 1) * np.log(pi + self.config.eps)
            ) + (self.config.beta_params[1] - 1) * np.log(1 - pi + self.config.eps)
            ll += beta_prior
            return -ll

        res = minimize_scalar(nll_beta_prior, bounds=(0.0, 1.0), method="bounded")
        return float(res.x)

    @gpu_dispatch(return_to_host=True)
    def _posterior_image(self, image: ArrayLike) -> np.ndarray:
        """Predict the NP posterior distribution of the given 2D image.

        Parameters
        ----------
        image : np.ndarray
            Grayscale 2D image to be analyzed.

        Returns
        -------
        np.ndarray
            The raw map of the posterior NP distribution, in range [0, 1].
        """

        hist_image = bincount_ubyte(image)
        pi = self._fit_pi(hist_image)

        mix = pi * self.p_np + (1 - pi) * self.p_bg
        posterior_bins = to_device((pi * self.p_np) / (mix + self.config.eps))

        posterior_map = posterior_bins[image]
        return posterior_map

    def predict(self, model_name: str | None = None) -> None:
        """Predict NP posterior map and save it under root.zarr/labels/nps/(spacing).
        Posterior is saved as float16"""

        # Save posteriors under root/labels/nps/(spacing)
        labels_group = self.stack_root.require_group("labels")
        seg_group = labels_group.require_group("nps")
        posterior_group = seg_group.require_group("posterior")

        compressor = BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle)
        posterior_arr = posterior_group.create_array(
            self.config.spacing_dir,
            shape=self.volume_stack.shape,
            chunks=(1, *self.volume_stack.shape[1:]),
            dtype="f2",
            compressors=compressor,
        )
        for idx in tqdm(
            range(self.volume_stack.shape[0]),
            desc="Calculating posterior",
            disable=not self.config.verbose,
        ):
            image = self.volume_stack[idx]
            posterior = self._posterior_image(image)
            posterior_arr[idx] = posterior

        processing = [
            {
                "step": "segmentation",
                "seg_target": "nps",
                "model": model_name,
                **self.config.to_serializable(),
            }
        ]

        posterior_arr.attrs["spacing"] = self.volume_stack.attrs["spacing"]
        posterior_arr.attrs["processing"] = (
            self.volume_stack.attrs["processing"] + processing
        )
        posterior_arr.attrs["inputs"] = self.volume_stack.path
        create_ome_multiscales(posterior_group)


@gpu_dispatch(return_to_host=True)
def bincount_ubyte(image: ArrayLike) -> np.ndarray:
    """Calculates image histogram with GPU acceleration"""
    return xp.bincount(image.ravel(), minlength=256).astype(xp.int64)
