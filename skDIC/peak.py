#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from skimage.feature import peak_local_max

def peak_prominent(image):
    """
    Return the prominent peak position.

    Parameters
    ----------
    image : (N, M) ndarray
        Input image.

    Returns
    -------
    peak : array of two elements.
    """
    peaks = peak_local_max(image, num_peaks=1, indices=True)
    return peaks[0]


def peak_near_center(image, halfwidth):
    """
    Return the peak position assuming that it's located
    near the center.

    This function seeks in a square centered at the image center.

    Parameters
    ----------
    image : (N, M) ndarray
        Input image.
    halfwidth : int
        Halfwidth of the square.

    Returns
    -------
    peak : array of two elements.
    """
    size = np.array(image.shape) // 2
    if (halfwidth > size).all():
        halfwidth = np.min(size)
    x0, y0 = size
    img = image[x0-halfwidth:x0+halfwidth+1,
                y0-halfwidth:y0+halfwidth+1]
    peaks = peak_local_max(img, num_peaks=1, indices=True)
    try:
        center_peak = np.array(peaks[0]) + (x0, y0) - halfwidth
    except IndexError:
        raise RuntimeError('Peak not detected near the center')
    return center_peak


def peak_near_center_fitted(image, halfwidth=3,
                            method='biparabolic', algorithm='leastsq'):
    """
    Find a peak near the center with a subpixel resolution.

    Peaks are fitted with a 2D curve.

    Parameters
    ----------
    image : ndarray
        Input image.
    halfwidth : int
        Halfwidth of the square. If None, the width is the image size.
    method : str, optional
        Available options are `biparabolic`, `quadrature` and `gaussian`.

    Returns
    -------
    peak : array of two elements.

    """
    # Detect roughtly the peak near the center
    if halfwidth is None:
        peaks = peak_local_max(image, num_peaks=1, indices=True)
        peaks = [np.array(peaks[0])]
        halfwidth = 3 # A small value for the fit... #FIXME
    else:
        peaks = [peak_near_center(image, halfwidth*2)]

    if algorithm == 'lmfit':
        import skDIC.peak_lmfit as find
    elif algorithm == 'curvefit':
        import skDIC.peak_curvefit as find
    elif algorithm == 'leastsq':
        import skDIC.peak_leastsq as find
    else:
        raise ValueError('Wrong algorithm keyword argument.')

    if method == 'biparabolic':
        return find.subpix_biparabolic(image, peaks, halfwidth=halfwidth)[0]
    elif method == 'quadrature':
        return find.subpix_quadrature(image, peaks, halfwidth=halfwidth)[0]
    elif method == 'gaussian':
        return find.subpix_gaussian(image, peaks, halfwidth=halfwidth)[0]
