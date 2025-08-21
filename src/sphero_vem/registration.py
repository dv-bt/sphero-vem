"""
This module contains functions used for the registration of volume stacks
"""

from enum import Enum
from pathlib import Path
from typing import Iterable, Callable
import SimpleITK as sitk
from sphero_vem.io import imwrite


class TransformType(Enum):
    SIMILARITY = "similarity"
    RIGID = "rigid"
    AFFINE = "affine"


def register_image_pair(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    transform_type: TransformType,
    shrink_factors: Iterable[int | float] = [1],
    smoothing_sigmas: Iterable[int | float] = [0.0],
    sampling_fractions: Iterable[float] | float = [1.0],
) -> tuple[sitk.Image, sitk.Transform]:
    """Register a pair of images and returns the calculated transformation. It uses
    gradient descent optimization on a mutual information criterion.
    Optimization is done with a multi resolution framework, but this can
    be turned off by passing a single value to shrink_factors."""

    # Transform
    transform_factory = get_transform_factory(transform_type)
    initial_transform = transform_factory()
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetInitialTransform(initial_transform)

    # Metric
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=30)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentagePerLevel(sampling_fractions)

    # Optimizer
    registration_method.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0, numberOfIterations=100
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Execute registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Apply the transform to the moving image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkBSpline3)
    resampler.SetTransform(final_transform)
    moving_image_final = resampler.Execute(moving_image)

    return moving_image_final, final_transform


def get_transform_factory(
    transform_type: TransformType,
) -> Callable[[], sitk.Transform]:
    """
    Returns a factory function for a specific transformation type.
    """
    transform_dispatcher = {
        TransformType.SIMILARITY: lambda: sitk.Similarity2DTransform(),
        TransformType.RIGID: lambda: sitk.Euler2DTransform(),
        TransformType.AFFINE: lambda: sitk.AffineTransform(2),
    }

    # Retrieve the function based on the Enum member
    return transform_dispatcher[transform_type]


def register_to_disk(
    fixed_image_path: Path,
    moving_image_path: Path,
    dest_path: Path,
    **registration_kwargs,
) -> None:
    """Read image pair, register them, and save the registered moving image"""
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    final_image, _ = register_image_pair(
        fixed_image, moving_image, **registration_kwargs
    )
    final_image = sitk.Cast(final_image, sitk.sitkUInt8)
    final_image_array = sitk.GetArrayViewFromImage(final_image)
    imwrite(dest_path, final_image_array)
