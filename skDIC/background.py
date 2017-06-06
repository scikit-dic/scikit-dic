#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from skimage import draw
from skimage.util import dtype_limits


dtypes_float = (np.float, np.float16, np.float32, np.float64)


def draw_circle_point(cy, cx, radius, dtype=np.uint8):
    """
    Draw a point made of concentric Andres' circles.

    Parameters
    ----------
    cy, cx : int
        Center coordinates of the point.
    radius : int
        Radius of the point.
    dtype : numpy dtype
        Data type used to generate the intensity values.

    Returns
    -------
    rr, cc, intensity : (N,) ndarray of int

    Notes
    -----
    The intensity decreases linearly along the radial position.

    Examples
    --------
    >>> img = np.zeros((100, 100), dtype=np.uint8)
    >>> points = draw_circle_point(30, 30, 5)
    >>> for rr, cc, intensity in points:
    ...     img[rr, cc] = intensity

    """
    imin, imax = dtype_limits(np.zeros((0,), dtype=dtype), clip_negative=False)
    if dtype in dtypes_float:
        intensity_step = (imax - imin) / radius
    else:
        intensity_step = (imax - imin) // radius

    data = []
    for rad in range(0, radius):
        pixels = draw.circle_perimeter(cy, cx, rad, method='andres')
        intensity = imax - rad * intensity_step
        intensity = intensity * np.ones(pixels[0].size)
        data.append((pixels[0], pixels[1], intensity))
    return data


def draw_square_point(cy, cx, halfwidth, dtype=np.uint8):
    """
    Draw a point made of concentric squares.

    Parameters
    ----------
    cy, cx : int
        Center coordinates of the point.
    halfwidth : int
        Half width of the point.
    dtype : numpy dtype
        Data type used to generate the intensity values.

    Returns
    -------
    rr, cc, intensity : (N,) ndarray of int

    Notes
    -----
    The intensity decreases linearly along the radial position.

    Examples
    --------
    >>> img = np.zeros((100, 100), dtype=np.uint8)
    >>> points = draw_square_point(30, 30, 5)
    >>> for rr, cc, intensity in points:
    ...     img[rr, cc] = intensity

    """
    imin, imax = dtype_limits(np.zeros((0,), dtype=dtype), clip_negative=False)
    if dtype in dtypes_float:
        intensity_step = (imax - imin) / halfwidth
    else:
        intensity_step = (imax - imin) // halfwidth

    data = []
    for pos in range(0, halfwidth):
        yi = [cy - pos, cy + pos, cy + pos, cy - pos]
        xi = [cx - pos, cx - pos, cx + pos, cx + pos]
        pixels = draw.polygon_perimeter(yi, xi)
        intensity = imax - pos * intensity_step
        intensity = intensity * np.ones(pixels[0].size)
        data.append((pixels[0], pixels[1], intensity))
    return data


def draw_points(size, num_points, radius, seed=1, func=draw_circle_point,
                dtype=np.uint8):
    """
    Draw points on the top of an image at random positions.

    Parameters
    ----------
    size : tuple of 2 elements
        Image size.
    num_points : int
        Number of points.
    radius : int
        Point radius.
    func : callable, optional
        Function to generate points. The function must return a
        tuple with point positions and intensities.
    seed : int, optional
        Seed used to generate random point positions.

    Returns
    -------
    image : (N, M) ndarray

    Examples
    --------
    >>> size = (100, 100)
    >>> img = draw_points(size, 100, 4, seed=1, func=draw_circle_point)

    """
    np.random.seed(seed)
    max_y, max_x = size
    imin, imax = dtype_limits(np.zeros((0,), dtype=dtype), clip_negative=False)
    img = np.ones(size, dtype=dtype) * imin

    points_x = np.random.randint(radius, high=max_x-1-radius, size=num_points)
    points_y = np.random.randint(radius, high=max_y-1-radius, size=num_points)
    for px, py in zip(points_x, points_y):
        points = func(py, px, radius, dtype=img.dtype)
        for rr, cc, intensity in points:
            img[rr, cc] = intensity
    return img
