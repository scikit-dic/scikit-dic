#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import optimize


def _residual_biparabolic(vars, xy, data, eps_data=1):
    x0 = vars[0]
    y0 = vars[1]
    c0 = vars[2]
    c1 = vars[3]

    x, y = xy
    model = c0 + c1 * (x - x0) ** 2 + c1 * (y - y0) ** 2

    return (data.ravel() - model.ravel()) / eps_data


def subpix_biparabolic(image, peaks, halfwidth=3):
    """
    Return the subpixel position of peaks with a biparabolic curve.

    The function has the form::

        c0 + c1 * (x - x0)**2 + c1 * (y - y0)**2

    Parameters
    ----------
    image : ndarray
        Input image.
    peaks :
        Pixel coordinates of pixel peaks.
    halfwidth : int, optional
        Number of pixels considered around the pixel peaks.

    Returns
    -------
    subpix_peaks : ndarray

    Notes
    -----
    This function is based on scipy.optimize.leastsq.

    """
    subpix_peaks = []
    for i, p in enumerate(peaks):
        px, py = p
        # halfwidth points on each side of the the peak
        # data.shape == (2*halfwidth+1, 2*halfwidth+1)
        data = image[px-halfwidth:px+halfwidth+1,
                     py-halfwidth:py+halfwidth+1]

        x = np.linspace(0, 2*halfwidth, 2*halfwidth+1)
        y = np.linspace(0, 2*halfwidth, 2*halfwidth+1)
        xy = np.meshgrid(x, y)

        initial_guess = (halfwidth, halfwidth, image[px, py], -1.)
        popt, pcov = optimize.leastsq(_residual_biparabolic, initial_guess,
                                      args=(xy, data))
        xpeak, ypeak = popt[0:2] + (px, py) - halfwidth
        subpix_peaks.append((xpeak, ypeak))
    return subpix_peaks


def _residual_quadrature(vars, xy, data, eps_data=1):
    x0 = vars[0]
    y0 = vars[1]
    c0 = vars[2]
    c1 = vars[3]
    c2 = vars[4]
    c3 = vars[5]
    c4 = vars[6]
    c5 = vars[7]

    x, y = xy
    x -= x0
    y -= y0
    model = c0 + c1 * x + c2 * y + c3 * x * y + c4 * x**2 + c5 * y**2

    return (data.ravel() - model.ravel()) / eps_data


def subpix_quadrature(image, peaks, halfwidth=3):
    """
    Return the subpixel position of peaks with a quadrature.

    The function has the form::

        c0 + c1 * x + c2 * y + c3 * x * y + c4 * x**2 + c5 * y**2

    Parameters
    ----------
    image : ndarray
        Input image.
    peaks :
        Pixel coordinates of pixel peaks.
    halfwidth : int, optional
        Number of pixels considered around the pixel peaks.

    Returns
    -------
    subpix_peaks : ndarray

    Notes
    -----
    This function is based on scipy.optimize.leastsq.

    """
    subpix_peaks = []
    for i, p in enumerate(peaks):
        px, py = p
        # halfwidth points on each side of the the peak
        # data.shape == (2*halfwidth+1, 2*halfwidth+1)
        data = image[px-halfwidth:px+halfwidth+1,
                     py-halfwidth:py+halfwidth+1]

        x = np.linspace(0, 2*halfwidth, 2*halfwidth+1)
        y = np.linspace(0, 2*halfwidth, 2*halfwidth+1)
        xy = np.meshgrid(x, y)

        initial_guess = (halfwidth, halfwidth, image[px, py], 0, 0, 0, -1., -1.)
        popt, pcov = optimize.leastsq(_residual_quadrature, initial_guess,
                                      args=(xy, data))
        xpeak, ypeak = popt[0:2] + (px, py) - halfwidth
        subpix_peaks.append((xpeak, ypeak))
    return subpix_peaks
