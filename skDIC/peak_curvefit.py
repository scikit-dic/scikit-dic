#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import optimize


def _twoD_gaussian(xy, x0, y0, amplitude, sigma_x, sigma_y, theta=0, offset=0):
    """
    Two dimensional Gaussian curve.

    Parameters
    ----------
    xy : ndarray
        Meshgrid.
    x0, y0 : float
        Gaussian center.
    amplitude : float
        Gaussian amplitude.
    sigma_x, sigma_y : float
        Gaussian width.
    theta : float, optional
        Orientation of the x-axis.
    offset : float, optional
        Vertical shift.

    Returns
    -------
    gaussian : ndarray

    """
    x, y = xy

    ctheta2 = np.cos(theta) ** 2
    stheta2 = np.sin(theta) ** 2
    s2theta = np.sin(2*theta)

    a = ctheta2 / (2*sigma_x**2) + stheta2 / (2*sigma_y**2)
    b = - s2theta / (4*sigma_x**2) + s2theta / (4*sigma_y**2)
    c = stheta2 / (2*sigma_x**2) + ctheta2 / (2*sigma_y**2)

    gaussian = np.exp(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))
    gaussian *= amplitude
    gaussian += offset
    return gaussian.ravel()


def _twoD_gaussian_symetric(xy, x0, y0, amplitude, sigma, offset=0):
    sigma_x = sigma
    sigma_y = sigma
    theta = 0
    return _twoD_gaussian(xy, x0, y0, amplitude, sigma_x, sigma_y, theta=theta,
                          offset=offset)


def _twoD_quadrature(xy, x0, y0, c0, c1, c2, c3, c4, c5):
    """
    Quadratic 2D curve.

    The function has the form::

        c0 + c1 * x + c2 * y + c3 * x * y + c4 * x**2 + c5 * y**2

    Parameters
    ----------
    xy : ndarray
        Meshgrid.
    c0, c1, c2, c3, c4, c5 : float
        Quadrature coefficients.

    Returns
    -------
    quadrature : ndarray

    References
    ----------
    .. [1] https://www.mathworks.com/matlabcentral/fileexchange/43073-improved-digital-image-correlation--dic-

    """
    x, y = xy
    x -= x0
    y -= y0
    quadrature = c0 + c1 * x + c2 * y + c3 * x * y + c4 * x**2 + c5 * y**2
    return quadrature.ravel()


def _twoD_biparabolic(xy, x0, y0, c0, c1):
    """
    Biparabolic 2D curve.

    The function has the form::

        c0 + c1 * (x - x0)**2 + c1 * (y - y0)**2

    Parameters
    ----------
    xy : ndarray
        Meshgrid.
    x0, y0 : float
        Parabola center.
    c0, c1 : float
        Parabola coefficients.

    Returns
    -------
    parabola : ndarray

    References
    ----------
    .. [1] D. J. Chen, F. P. Chiang, Y. S. Tan, and H. S. Don
       "Digital speckle-displacement measurement using a complex spectrum method"
       Applied Optics, 32, 11, pp. 1839-1849 (1993).
    """
    x, y = xy
    parabola = c0 + c1 * (x - x0)**2 + c1 * (y - y0)**2
    return parabola.ravel()


def subpix_biparabolic(image, peaks, halfwidth=2):
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
    This function is based on scipy.optimize.curvefit.

    """
    subpix_peaks = []
    for px, py in peaks:
        # halfwidth points on each side of the the peak
        # data.shape == (2*halfwidth+1, 2*halfwidth+1)
        data = image[px-halfwidth:px+halfwidth+1,
                     py-halfwidth:py+halfwidth+1]

        x = np.linspace(0, 2*halfwidth, 2*halfwidth+1)
        y = np.linspace(0, 2*halfwidth, 2*halfwidth+1)
        x, y = np.meshgrid(x, y)

        bounds = ((0, 0, image[px, py] / 2., -np.inf),
                  (2*halfwidth, 2*halfwidth, 2 * image[px, py], 0))
        initial_guess = (halfwidth, halfwidth, image[px, py], -1.)
        # Scipy's doc says that dogbox is suitable for small problems with bounds
        popt, pcov = optimize.curve_fit(_twoD_biparabolic, (x, y), data.ravel(),
                                        p0=initial_guess,
                                        bounds=bounds,
                                        method='dogbox')
        xpeak, ypeak = popt[0:2] + (px, py) - halfwidth
        subpix_peaks.append((xpeak, ypeak))
    return subpix_peaks


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
    This function is based on scipy.optimize.curvefit.

    """
    subpix_peaks = []
    for px, py in peaks:
        # halfwidth points on each side of the the peak
        # data.shape == (2*halfwidth+1, 2*halfwidth+1)
        data = image[px-halfwidth:px+halfwidth+1,
                     py-halfwidth:py+halfwidth+1]

        x = np.linspace(0, 2*halfwidth, 2*halfwidth+1)
        y = np.linspace(0, 2*halfwidth, 2*halfwidth+1)
        x, y = np.meshgrid(x, y)

        initial_guess = (halfwidth, halfwidth, image[px, py], 0., 0., 0., -1., -1.)
        bounds = ((0, 0, image[px, py] / 2., -np.inf, -np.inf, -np.inf, -np.inf, -np.inf),
                  (2*halfwidth, 2*halfwidth, 2 * image[px, py], np.inf, np.inf, np.inf, 0, 0))
        popt, pcov = optimize.curve_fit(_twoD_quadrature, (x, y), data.ravel(),
                                        p0=initial_guess,
                                        bounds=bounds,
                                        method='dogbox')
        xpeak, ypeak = popt[0:2] + (px, py) - halfwidth
        subpix_peaks.append((xpeak, ypeak))
    return subpix_peaks


def subpix_gaussian(image, peaks, halfwidth=5, sigmas=3):
    """
    Return the subpixel position of peaks with a Gaussian.

    Parameters
    ----------
    image : ndarray
        Input image.
    peaks :
        Pixel coordinates of pixel peaks.
    halfwidth : int, optional
        Number of pixels considered around the pixel peaks.
    sigmas : int, optional
        Gaussian width.

    Returns
    -------
    subpix_peaks : ndarray

    """
    subpix_peaks = []
    for px, py in peaks:
        # halfwidth points on each side of the the peak
        # data.shape == (2*halfwidth+1, 2*halfwidth+1)
        data = image[px-halfwidth:px+halfwidth+1,
                     py-halfwidth:py+halfwidth+1]

        x = np.linspace(0, 2*halfwidth, 2*halfwidth+1)
        y = np.linspace(0, 2*halfwidth, 2*halfwidth+1)
        x, y = np.meshgrid(x, y)

        #initial_guess = (halfwidth, halfwidth, image[px, py], sigmas, sigmas, 0, 0)
        initial_guess = (halfwidth, halfwidth, image[px, py], sigmas, 0)
        bounds = ((0, 0, image[px, py] /2., 1, 0),
                  (2*halfwidth, 2*halfwidth, 2 * image[px, py], 2*halfwidth, image[px, py]))
        popt, pcov = optimize.curve_fit(_twoD_gaussian_symetric, (x, y), data.ravel(),
                                        bounds=bounds,
                                        method='dogbox',
                                        p0=initial_guess)

        xpeak, ypeak = popt[0:2] + (px, py) - halfwidth
        subpix_peaks.append((xpeak, ypeak))
    return subpix_peaks
