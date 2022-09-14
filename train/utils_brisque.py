import piq
import math
import torch
import typing
import warnings

import numpy as np
import torch.nn.functional as F

from typing import Union, Tuple
from piq.utils import _validate_input, _reduce
from piq.functional import rgb2yiq, gaussian_filter, imresize
from piq.brisque import _ggd_parameters, _aggd_parameters, _natural_scene_statistics, _scale_features



def brisque_core(x: torch.Tensor,
            kernel_size: int = 7, kernel_sigma: float = 7 / 6,
            data_range = 1., reduction: str = 'mean') -> torch.Tensor:
    r"""Interface of BRISQUE index.
    Supports greyscale and colour images with RGB channel order.
    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``
    Returns:
        Value of BRISQUE index.
    """
    assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'
    _validate_input([x, ], dim_range=(4, 4), data_range=(0, data_range))

    x = x / float(data_range) * 255

    if x.size(1) == 3:
        # Mimic matlab rgb2gray, which keeps image in uint8 during colour conversion
        x = torch.round(rgb2yiq(x)[:, :1])
    features = []
    num_of_scales = 2
    for _ in range(num_of_scales):
        features.append(_natural_scene_statistics(x, kernel_size, kernel_sigma))
        x = imresize(x, sizes=(x.size(2) // 2, x.size(3) // 2))

    features = torch.cat(features, dim=-1)
    # nrm_features = (features + features.min().abs()) / (features.max() - features.min())
    # scaled_features = _scale_features(features)
    
    return features


def brisque(image: np.array,
            kernel_size: int = 7, kernel_sigma: float = 7 / 6,
            data_range = 1., reduction: str = 'mean') -> torch.Tensor:
    x = torch.tensor(image)
    x = x / 255

    x = x.reshape((1, x.shape[2],x.shape[0],x.shape[1]))

    return(np.array(brisque_core(x)))
    