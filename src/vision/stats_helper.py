import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################

    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    n = 0
    img_paths = glob.glob(os.path.join(dir_name, "**", "*.jpg"), recursive=True)
    for path in img_paths:
        with Image.open(path) as im:
            img = np.asarray(im.convert("L"), dtype=np.float64) / 255.0
        pixel_sum += img.sum()
        pixel_sq_sum += (img ** 2).sum()
        n += img.size

    mean = pixel_sum / n
    variance = (pixel_sq_sum - n * mean ** 2) / (n - 1)
    std = np.sqrt(variance)

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
