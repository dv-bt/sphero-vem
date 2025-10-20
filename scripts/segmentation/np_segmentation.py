"""
Nanoparticle segmentation
"""

from pathlib import Path
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np
from tifffile import imread
from scipy.integrate import simpson
from scipy.optimize import minimize_scalar
from sphero_vem.io import write_image
from sphero_vem.utils import generate_manifest


@dataclass
class NanoparticleConfig:
    """Configuration class for NP segmentation"""

    patches_root: Path
    verbose: bool = False
    max_iter: int = 10000
    w_bg_init: float = 0.5
    eps: float = 1e-12
    tol: float = 1e-8
    posterior_threshold = 0.9

    bg_patches: list[Path] = field(init=False)
    np_patches: list[Path] = field(init=False)

    def __post_init__(self) -> None:
        """Get list of patches"""
        self.bg_patches = list(patches_root.glob("bg-crops/*.tif"))
        self.np_patches = list(patches_root.glob("np-crops/*.tif"))


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
        self.bg_patches = self.config.bg_patches
        self.np_patches = self.config.np_patches

    def fit(self) -> None:
        """Find the NP probability density"""
        p_mix = self._calc_patch_prob(self.np_patches)
        self.p_bg = self._calc_patch_prob(self.bg_patches)

        # Run EM with multiple restarts
        results = []
        for w_bg_init in tqdm(
            np.linspace(0.1, 0.9, 9),
            "Running EM algorithm with multiple restarts",
            disable=not config.verbose,
        ):
            results.append((w_bg_init, *self._deconvolve_mixture(p_mix, w_bg_init)))

        # Calculate best initialization value
        best = sorted(results, key=lambda x: x[2], reverse=True)[0]
        self.w_bg_init: np.float64 = best[0]
        self.p_np: np.ndarray = best[1]
        self.log_likelihood: np.float64 = best[2]

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the NP posterior distribution of the given image.

        Parameters
        ----------
        image : np.ndarray
            Grayscale 2D image to be analyzed.

        Returns
        -------
        np.ndarray
            The posterior maps thresholded at the value specified by config.
        np.ndarray
            The raw map of the posterior NP distribution, in range [0, 1].
        """

        def residual_loss(pi_np, p_image) -> float:
            """Sum of squared residuals between the image probability and the estimated
            mixture probability."""
            mix_est = pi_np * self.p_np + (1 - pi_np) * self.p_bg
            return np.sum((p_image - mix_est) ** 2)

        x = np.linspace(0, 255, 256)
        p_image, _ = np.histogram(image, bins=256, range=(0, 256), density=True)
        res = minimize_scalar(
            residual_loss, bounds=(0, 1), method="bounded", args=p_image
        )
        pi_np = res.x

        posterior = (pi_np * self.p_np) / (
            pi_np * self.p_np + (1 - pi_np) * self.p_bg + self.config.eps
        )
        posterior_LUT = np.interp(np.arange(256), x, posterior)
        posterior_map = posterior_LUT[image]
        posterior_th = (posterior_map > self.config.posterior_threshold) * 1

        return posterior_th, posterior_map

    def _calc_patch_prob(self, patches: list[Path]) -> np.ndarray:
        """Calculate probability density from list of patches"""
        patch_list = []
        for path in patches:
            patch_list.append(imread(path))

        patch_vector = np.concatenate([patch.flatten() for patch in patch_list])
        p_patch, _ = np.histogram(patch_vector, bins=256, range=(0, 256), density=True)
        return p_patch

    def _deconvolve_mixture(
        self, p_mix: np.ndarray, w_bg_init: float = 0.5
    ) -> tuple[np.ndarray, float]:
        """
        EM decomposition of p_mix = w_bg * p_bg + w_np * p_np.

        Parameters
        ----------
        p_mix : np.ndarray
            Probability density function of the mixture.
        w_bg_init : float
            Initialization value for the background weight in the mixture

        Returns
        -------
        np.ndarray
            The deconvolved NP probability density.
        np.float64
            Data log likelihood after fitting.
        """
        # Initialize variables
        w_bg = w_bg_init
        w_np = 1.0 - w_bg
        x = np.linspace(0, 255, 256)
        eps = self.config.eps

        # Initialize p_np
        p_np = np.clip((p_mix - w_bg * self.p_bg) / w_np, eps, None)
        p_np /= simpson(p_np, x)

        def calc_log_mix() -> np.ndarray:
            """Calculate log mixture probability"""
            log_wb_pb = np.log(w_bg + eps) + np.log(self.p_bg + eps)
            log_wo_po = np.log(w_np + eps) + np.log(p_np + eps)
            return np.logaddexp(log_wb_pb, log_wo_po)

        def calc_gamma() -> np.ndarray:
            """Calculate background responsibility"""
            num = np.log(w_bg + eps) + np.log(self.p_bg + eps)
            den = calc_log_mix()
            return np.exp(num - den)

        def calc_log_likelihood() -> np.float64:
            """Calculate data log likelihood"""
            return simpson(p_mix * calc_log_mix(), x).item()

        for i in range(self.config.max_iter):
            # E-step
            gamma = calc_gamma()

            # M-step
            w_bg_new = simpson(gamma * p_mix, x)
            w_np_new = 1.0 - w_bg_new
            p_np_new = (1.0 - gamma) * p_mix / (w_np_new + eps)
            p_np_new = np.clip(p_np_new, eps, None)
            p_np_new /= simpson(p_np_new, x)

            # Convergence check
            if abs(w_bg_new - w_bg) < self.config.tol:
                break

            # Update for next iteration
            w_bg, w_np = w_bg_new, w_np_new
            p_np = p_np_new

        return p_np, calc_log_likelihood()


if __name__ == "__main__":
    patches_root = Path("data/processed/labeled/Au_01-vol_01/labeled-01")
    image_dir = Path("data/processed/aligned/Au_01-vol_01")
    pred_dir = Path("data/processed/segmented/Au_01-vol_01/nps")
    pred_dir.mkdir(exist_ok=True, parents=True)
    image_list = sorted(image_dir.glob("*.tif"))

    config = NanoparticleConfig(patches_root, verbose=True)
    segmentation = NanoparticleSegmentation(config)
    segmentation.fit()

    for image_path in tqdm(image_list, "Analyzing images"):
        image = imread(image_path)
        posterior_th, _ = segmentation.predict(image)
        write_image(
            pred_dir / f"{image_path.name}-nps.tif",
            posterior_th,
            compressed=True,
        )

    generate_manifest(
        dataset=image_dir.name,
        out_dir=pred_dir,
        images=image_list,
        processing=[
            {
                "step": "segmentation",
                "seg_target": "nps",
                "w_bg_init": float(segmentation.w_bg_init),
                "max_iter": segmentation.config.max_iter,
                "tol": segmentation.config.tol,
                "log_likelihood": float(segmentation.log_likelihood),
                "posterior_threshold": segmentation.config.posterior_threshold,
                "bg_patches": [str(i) for i in segmentation.config.bg_patches],
                "np_patches": [str(i) for i in segmentation.config.np_patches],
            }
        ],
    )
