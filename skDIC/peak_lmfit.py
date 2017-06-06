#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from lmfit import minimize, Parameters, report_fit


def _residual_biparabolic(Params, xy, data, eps_data=1):
    x0 = Params['x0'].value
    y0 = Params['y0'].value
    c0 = Params['c0'].value
    c1 = Params['c1'].value

    x, y = xy
    model = c0 + c1 * (x - x0) ** 2 + c1 * (y - y0) ** 2

    return (data - model) / eps_data


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
    This function is based on lmfit.

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
        params = Parameters()
        params.add('x0', value=halfwidth, min=halfwidth-1, max=halfwidth+1)
        params.add('y0', value=halfwidth, min=halfwidth-1, max=halfwidth+1)
        params.add('c0', value=image[px, py], min=image[px, py] / 2., max=image[px, py])
        params.add('c1', value=-1.)


        out = minimize(_residual_biparabolic, params, args=(xy, data))
        res = np.array((out.params['x0'].value, out.params['y0'].value))
        #report_fit(out.params)
        xpeak, ypeak = res + (px, py) - halfwidth
        subpix_peaks.append((xpeak, ypeak))
    return subpix_peaks


def _residual_quadrature(Params, xy, data, eps_data=1):
    x0 = Params['x0'].value
    y0 = Params['y0'].value
    c0 = Params['c0'].value
    c1 = Params['c1'].value
    c2 = Params['c2'].value
    c3 = Params['c3'].value
    c4 = Params['c4'].value
    c5 = Params['c5'].value

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
    This function is based on lmfit.

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
        params = Parameters()
        params.add('x0', value=halfwidth, min=halfwidth-1, max=halfwidth+1)
        params.add('y0', value=halfwidth, min=halfwidth-1, max=halfwidth+1)
        params.add('c0', value=image[px, py], min=0)
        params.add('c1', value=0, vary=False)
        params.add('c2', value=0, vary=False)
        params.add('c3', value=0, vary=False)
        params.add('c4', value=-1., max=0)
        params.add('c5', value=-1., max=0)


        out = minimize(_residual_quadrature, params, args=(xy, data))
        res = np.array((out.params['x0'].value, out.params['y0'].value))
        #report_fit(out.params)
        xpeak, ypeak = res + (px, py) - halfwidth
        subpix_peaks.append((xpeak, ypeak))
    return subpix_peaks
