"""
Nanoparticle segmentation

TODO: standardize spacing vs spacing_dir input
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
import skimage as ski_cpu
from sphero_vem.utils import (
    CustomJSONEncoder,
    create_ome_multiscales,
    dirname_from_spacing,
)
from sphero_vem.utils.accelerator import (
    xp,
    gpu_dispatch,
    ArrayLike,
    ndi,
    to_host,
    to_device,
)
from sphero_vem.io import write_zarr
from sphero_vem.postprocessing import binary_closing, filter_and_relabel
from sphero_vem.segmentation.np.utils import bincount_ubyte


@dataclass
class NanoparticleSegConfig:
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
    zarr_chunks: tuple[int, int, int] = (1, 1024, 1024)

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
        config: NanoparticleSegConfig,
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
        self.volume_stack: zarr.Array = self.stack_root.get(
            f"images/{self.config.spacing_dir}"
        )

    @classmethod
    def load(cls, model_dir: str | Path) -> Self:
        """Load a pretrained model from a directory. The directory is expected to have
        a training_manifest.json and model_params.npz of the correct format"""
        if isinstance(model_dir, str):
            model_dir = Path(model_dir)
        config = NanoparticleSegConfig.from_json(model_dir / "training_manifest.json")
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
        """Normalize a count histogram to a probability mass function.

        Converts raw counts to a PMF and adds a small floor (``config.eps``)
        to every bin to avoid division by zero in log-likelihood calculations.

        Parameters
        ----------
        hist : numpy.ndarray
            Integer count histogram of length 256.

        Returns
        -------
        numpy.ndarray
            Normalized PMF of length 256, dtype float64.
        """
        pmf = hist.astype(np.float64)
        if pmf.sum() == 0:
            return np.full(256, 1 / 256, dtype=np.float64)
        pmf /= pmf.sum()
        # Add floor and re-normalize
        pmf = np.maximum(pmf, self.config.eps)
        pmf /= pmf.sum()
        return pmf

    def _calc_stack_hist(self) -> None:
        """Compute and store the intensity histogram of the full image stack.

        Iterates over slices at intervals of ``config.sampling_step``, accumulates
        a 256-bin uint8 histogram, normalizes it to a PMF, and stores the results
        in ``self.hist_stack`` and ``self.p_stack``.
        """
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
        """Compute and store the intensity histogram of background pixels.

        Determines intensity thresholds from the stack PMF, then accumulates
        background pixel counts across sampled slices using ``_extract_bg_hist``.
        Stores results in ``self.hist_bg`` and ``self.p_bg``.
        """

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
        """Extract the background pixel histogram from a single 2D slice.

        Identifies candidate NP regions by intensity thresholding, expands their
        bounding boxes by a halo, and counts intensity values only in pixels
        outside those regions (background). Returns None if no background pixels
        are found.

        Parameters
        ----------
        image : ArrayLike
            Grayscale 2D image slice (uint8).

        Returns
        -------
        ArrayLike | None
            256-bin integer histogram of background pixels, or None if no
            background was found in this slice.
        """

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
        """Expand a ``scipy.ndimage.find_objects`` bounding box by a halo margin.

        Parameters
        ----------
        bbox : tuple[slice]
            Bounding box as returned by ``find_objects`` — a tuple of two
            slices (Y, X).
        image_shape : tuple
            Shape of the 2D image, used to clip the expanded bbox to valid bounds.

        Returns
        -------
        tuple[slice]
            Expanded bounding box clipped to image boundaries.
        """
        (slice_y, slice_x) = bbox
        y0 = max(slice_y.start - self.config.halo_pad, 0)
        y1 = min(slice_y.stop + self.config.halo_pad, image_shape[0])
        x0 = max(slice_x.start - self.config.halo_pad, 0)
        x1 = min(slice_x.stop + self.config.halo_pad, image_shape[1])
        return (slice(y0, y1), slice(x0, x1))

    def _init_w_bg(self) -> np.float64:
        """Initialize the background mixing weight for the EM algorithm.

        Returns
        -------
        numpy.float64
            Initial ``w_bg`` estimate: ratio of background pixel count to total
            pixel count across the sampled stack.
        """
        return self.hist_bg.sum() / self.hist_stack.sum()

    def _init_p_np(self) -> np.ndarray:
        """Initialize the NP probability distribution for the EM algorithm.

        Sets bins below the low-intensity threshold to zero and normalizes the
        tail of the stack distribution to serve as the initial NP PMF.

        Returns
        -------
        numpy.ndarray
            Initial NP PMF of length 256, concentrated on high-intensity bins.
        """
        p_np = np.zeros_like(self.p_stack, dtype=np.float64)
        p_np[self.th_low :] = self.p_stack[self.th_low :]
        return self._normalize_pmf(p_np)

    def _deconvolve_mixture(self):
        """Fit the two-component intensity mixture model via EM.

        Runs a plain EM algorithm that alternates between computing posterior
        responsibilities (E-step) and updating the NP PMF and background weight
        by minimizing the negative log-likelihood (M-step). Stores the fitted
        NP PMF in ``self.p_np`` and convergence summary in ``self.summary_fit``.
        """

        # Initialization
        w_bg = self._init_w_bg()
        p_np = self._init_p_np()

        def neg_log_likelihood(w_bg: float, p_np: np.ndarray) -> np.float64:
            """Calculate observed negative data log likelihood"""
            mix = w_bg * self.p_bg + (1 - w_bg) * p_np
            return -np.sum(self.p_stack * np.log(mix + self.config.eps))

        nll_prev = np.inf
        for i in range(self.config.max_iter):
            # E-step
            num_bg = w_bg * self.p_bg
            num_np = (1 - w_bg) * p_np
            gamma_bg = num_bg / (num_bg + num_np + self.config.eps)

            # M-step
            q = (1.0 - gamma_bg) * self.p_stack
            q_sum = q.sum()
            if q.sum() > self.config.eps:
                p_np = self._normalize_pmf(q)
                w_bg = np.sum(self.p_stack * gamma_bg)
                w_bg = np.clip(w_bg, self.config.eps, 1 - self.config.eps)
            else:
                raise RuntimeError(
                    f"EM collapsed at iteration {i}: foreground responsibility "
                    f"vanished (q_sum={q_sum:.2e}). Check initialization or input data."
                )

            nll = neg_log_likelihood(w_bg, p_np)
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
        """Estimate the per-image NP mixing weight by MAP optimization.

        Minimizes the negative log-likelihood of the observed pixel histogram
        under the fitted mixture model, regularized by a Beta prior on ``pi``.

        Parameters
        ----------
        hist_image : numpy.ndarray
            256-bin integer histogram of the image to be analyzed.

        Returns
        -------
        float
            MAP estimate of the NP mixing weight ``pi`` in [0, 1].
        """

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
    def _posterior_image(self, image: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Predict the NP posterior distribution of the given 2D image.

        Parameters
        ----------
        image : np.ndarray
            Grayscale 2D image to be analyzed.

        Returns
        -------
        np.ndarray
            The raw map of the posterior NP distribution, in range [0, 1].
        np.ndarray
            The NP posterior distribution, in range [0, 1].
        """

        hist_image = bincount_ubyte(image)
        pi = self._fit_pi(hist_image)

        mix = pi * self.p_np + (1 - pi) * self.p_bg
        posterior_bins = to_device((pi * self.p_np) / (mix + self.config.eps))

        posterior_map = posterior_bins[image]
        return posterior_map, posterior_bins

    def predict(self, model_name: str | None = None) -> None:
        """Predict NP posterior map and save it under root.zarr/labels/nps/(spacing).
        Posterior is saved as float16"""

        # Save posteriors under root/labels/nps/(spacing)
        posterior_group = self.stack_root.require_group("labels/nps/posterior")

        compressor = BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle)
        posterior_arr = posterior_group.create_array(
            self.config.spacing_dir,
            shape=self.volume_stack.shape,
            chunks=self.config.zarr_chunks,
            dtype="f2",
            compressors=compressor,
        )
        for idx in tqdm(
            range(self.volume_stack.shape[0]),
            desc="Calculating posterior",
            disable=not self.config.verbose,
        ):
            image = self.volume_stack[idx]
            posterior, _ = self._posterior_image(image)
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


def label_nanoparticles(
    root_path: Path,
    spacing: tuple[int, int, int],
    threshold: float,
    radius: int = 1,
    connectivity: int = 2,
    min_size: int = 10,
) -> None:
    """Threshold posterior >= threshold and label nanoparticle binary masks.

    Masks are first subject to a binary closing with given radius. Then, labeling is
    done with the specified connectivity and labels smaller than min_size (in voxels)
    are discarded. The volume is finally relabeled to have sequential label IDs.

    Parameters
    ----------
    root_path : Path
        Path to the root of the zarr store
    spacing : tuple[int, int, int]
        Spacing to use for the labeling
    threshold : float
        Theshold to apply to the nanoparticle posterior, such that posterior >= threshold.
        It should be between 0 and 1.
    radius : int
        Radius in voxels for a ball element using during binary closing.
        Default is 1.
    connectivity : int
        Connectivity used during labeling. Default is 2.
    min_size : int
        Minimimum label size in voxels. Labels with size < min_size will be discarded.
        Default is 10.

    """

    root = zarr.open_group(root_path)
    spacing_dir = dirname_from_spacing(spacing)

    posterior_zarr: zarr.Array = root.get(f"labels/nps/posterior/{spacing_dir}")

    # Threshold posterior
    posterior = posterior_zarr[:]
    masks = posterior >= threshold

    # Label masks
    masks_filt = binary_closing(masks, radius=radius)
    masks_filt = ski_cpu.measure.label(masks_filt, connectivity=connectivity)
    masks_filt = filter_and_relabel(masks_filt, min_size=min_size)

    write_zarr(
        root,
        masks_filt,
        dst_path=f"labels/nps/masks/{spacing_dir}",
        src_zarr=posterior_zarr,
        processing={
            "step": "labeling",
            "radius": radius,
            "connectivity": connectivity,
            "min_size": min_size,
        },
    )
