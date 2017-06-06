#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import Rbf, interp2d, RectBivariateSpline
from scipy.stats import entropy as scipy_entropy
from itertools import tee


def pairwise(iterable):
    """
    Generate an enumerated pairwise from an interable.


    s -> (0, (s0,s1)), (1, (s1,s2)), (2, (s2, s3)), ...

    Parameters
    ----------
    interable : iterable
        An iterable.

    Returns
    -------
    it : iterable

    References
    ----------
    .. [1] https://docs.python.org/3/library/itertools.html

    """
    a, b = tee(iterable)
    next(b, None)
    return enumerate(zip(a, b))


def entropy(img, base=2):
    """
    Calculate the entropy of a grayscale image.

    Parameters
    ----------
    img : (N, M) ndarray
        Grayscale input image.
    base : float, optional
        The logarithmic base to use.

    Returns
    -------
    entropy : float

    Notes
    -----
    The units are bits or shannon (Sh) for base=2, natural unit (nat) for
    base=np.e and hartley (Hart) for base=10.
    """
    return scipy_entropy(img.ravel(), base=base)


def interpolate_image(img, num_points=2, method='RectBivariateSpline',
                      kind='quadratic'):
    """
    Interpolate an image.

    Parameters
    ----------
    img : (N, M) ndarray
        Input image.
    num_points : int, optional
        Number of points introduced between fixed data points.
    method : str, optional
        Method used to interpolate. Must be `rbf`, `interp2d` or
        `RectBivariateSpline`.
    kind : str, optional
        Function used to interpolate. For valid parameters, see
        the `kind` argument of `np.interpolate.interp2d` or
        the `function` argument of `np.interpolate.Rbf`.

    Returns
    -------
    interp_img : (N*num_points, M*num_points) ndarray

    Notes
    -----
    For `RectBivariateSpline`, available kinds are: `linear`,
    `quadratic`, `cubic`.
    """
    # Data points for interpolation
    x, y = np.indices(img.shape)
    z = np.ravel(img)
    # Interpolation grid
    num_points += 1  # need to increment for linspace
    ti = np.linspace(0, img.shape[0], num_points*img.shape[0], endpoint=False)
    tj = np.linspace(0, img.shape[1], num_points*img.shape[1], endpoint=False)
    # Interpolate
    Xi, Yi = np.meshgrid(ti, tj)
    if method.lower() == 'rbf':
        function = Rbf(x, y, z, epsilon=2, function=kind)
        Zi = function(Xi, Yi)
    elif method.lower() == 'interp2d':
        function = interp2d(x, y, z, kind=kind, fill_value=0)
        Zi = function(ti, tj)
    elif method.lower() == 'rectbivariatespline':
        if kind.lower() == 'linear':
            k = 1
        elif kind.lower() == 'quadratic':
            k = 2
        elif kind.lower() == 'cubic':
            k = 3
        else:
            raise ValueError('Wrong kind')
        xi, yi = x[:, 0], y[0, :]
        function = RectBivariateSpline(yi, xi, img.T, kx=k, ky=k)
        Zi = function(tj, ti)
    else:
        raise ValueError('Wrong method')
    return Zi.reshape((tj.size, ti.size)).T


def interpolate_nan(img):
    """
    Interpolate NaN values in an image.

    Parameters
    ----------
    image : (N, M) ndarray
        Input image.

    Returns
    -------
    output : (N, M) ndarray
    """
    image = img.copy()
    nans = np.isnan(image)
    # Indices to interpolate
    xi, yi = nans.nonzero()
    notnans = ~nans
    # Known values
    x, y = notnans.nonzero()
    z = np.ravel(image[~nans])

    ff = Rbf(x, y, z, function='linear')
    for i, j in zip(xi, yi):
        image[i, j] = ff(i, j)
    return image
