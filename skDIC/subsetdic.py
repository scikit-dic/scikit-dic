#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import numpy as np


from skimage.feature import match_template
from skimage.feature.peak import peak_local_max
from skimage._shared._warnings import warn

from .peak import peak_prominent, peak_near_center, peak_near_center_fitted
from .utils import interpolate_image
from ._dft import register_translation


def extract_subimg(img1, img2, coord, bsize=16, seek=None):
    """
    Extract subimages at a given location.

    This function is provided to help the user to test elementary
    subset DIC algorithms on images of interests.

    Parameters
    ----------
    img1 : (N, M) ndarray
        Reference image.
    img2 : (N, M) ndarray
        Deformed image.
    coord : tuple
        Coordinates of the origin point.
    bsize : int, optional
        Box size.
    seek : int, optional
        Number of pixels considered to seek the displacement.
        If `None`, the value is set to `bsize`.

    Returns
    -------
    (subimg1, subimg2) : tuple of ndarray

    """
    # Number of pixels considered to seek the displacement.
    if seek is None:
        seek = bsize

    i, j = coord
    subimg1 = img1[i:i+bsize, j:j+bsize]
    subimg2 = img2[i-seek:i+bsize+seek, j-seek:j+bsize+seek]
    return subimg1, subimg2


def pixelar(subimg1, subimg2, correlator='ZNCC'):
    """
    Calculate the displacement field between subimg1 and subimg2
    with a pixel-size resolution.

    Parameters
    ----------
    subimg1 : (N, N) ndarray
        Reference image.
    subimg2 : (M, M) ndarray
        Deformed image, M >= N.
    correlator : str, optional
        Method for cross-correlation computation.
        Valid names are ZNCC or DFT.

    Returns
    -------
    result : dict
        The keys of the dictionnary are:
            * 'CCimage_pix': Image of the pixellar cross-correlation
            * 'displacement_pix': Pixelar displacement
            * 'CCimage': Cross-correlation at the best refinement
            * 'displacement': Cross-correlation at the best refinement

    Notes
    -----
    The ZNCC correlator uses `skimage.feature.match_template`.
    The DFT correlator is a modified version of `skimage.feature.register_translation`.

    References
    ----------
    .. [1] Briechle and Hanebeck, "Template Matching using Fast Normalized
           Cross Correlation", Proceedings of the SPIE (2001).
           DOI:10.1117/12.421129
    .. [2] J. P. Lewis, "Fast Normalized Cross-Correlation", Industrial Light
           and Magic.
    .. [3] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           “Efficient subpixel image registration algorithms,” Optics Letters
           33, 156-158 (2008). DOI:10.1364/OL.33.000156

    Examples
    --------
    >>> from skimage import transform as tf
    >>> img1 = np.random.normal(size=(200, 200))
    >>> tform = tf.AffineTransform(scale=(1, 1), rotation=0,
                                   translation=(3, 2), shear=0.03)
    >>> img2 = tf.warp(img1, tform)
    >>> coord = 50, 50
    >>> subimg1, subimg2 = extract_subimg(img1, img2, coord)
    >>> res = pixelar(subimg1, subimg2)

    """
    # Check images are squares
    for si in (subimg1, subimg2):
        if si.shape[0] != si.shape[1]:
            raise ValueError('Images must have a dimension (N, N).')
    # Check that dimensions of img2 larger than img1
    if subimg2.shape[0] < subimg1.shape[0]:
            raise ValueError('subimg2 must have a dimension equal or larger than subimg1.')

    if correlator.lower() == 'zncc':
        pix_correl = match_template(subimg2, subimg1, pad_input=True)
        pix_peak = peak_local_max(pix_correl, num_peaks=1, indices=True)[0]
        pix_displacement = np.array(pix_peak - np.array(pix_correl.shape) / 2.,
                                    dtype=np.int).tolist()
    elif correlator.lower() == 'dft':
        # Images must have the same size, we crop.
        offset = int((subimg2.shape[0] - subimg1.shape[0]) / 2.)
        subimg2 = subimg2[offset:subimg1.shape[0]+offset, offset:subimg1.shape[0]+offset]
        pix_displacement, error, diffphase, pix_correl = register_translation(subimg2, subimg1,
                                                                              upsample_factor=1, space='real')
    else:
        raise ValueError('Wrong name for the correlator option.')

    return {'CCimage_pix': pix_correl,
            'displacement_pix': pix_displacement,
            'CCimage': pix_correl,
            'displacement': pix_displacement}


def subpixelar_dft(subimg1, subimg2, upsample_factor=10):
    """
    Calculate the displacement field between subimg1 and subimg2.

    Parameters
    ----------
    subimg1 : (N, N) ndarray
        Reference image.
    subimg2 : (M, M) ndarray
        Deformed image, M >= N.

    Returns
    -------
    result : dict
        The keys of the dictionnary are:
            * 'CCimage_pix': Image of the pixellar cross-correlation
            * 'displacement_pix': Pixelar displacement
            * 'CCimage_subpix': Image of the subpixellar cross-correlation
            * 'displacement_subpix': Subpixelar displacement
            * 'CCimage': Cross-correlation at the best refinement
            * 'displacement': Cross-correlation at the best refinement

    Notes
    -----
    subimg2 is croped to ensure that both images have the same shape,
    since DFT requires identical shapes.

    References
    ----------
    .. [1] Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           “Efficient subpixel image registration algorithms,” Optics Letters
           33, 156-158 (2008). DOI:10.1364/OL.33.000156
    """
    # Images must have the same size, we crop.
    offset = int((subimg2.shape[0] - subimg1.shape[0]) / 2.)
    subimg2 = subimg2[offset:subimg1.shape[0]+offset, offset:subimg1.shape[0]+offset]
    subpix_dis, error, diffphase, subpix_correl = register_translation(subimg2, subimg1,
                                                                       upsample_factor=upsample_factor,
                                                                       space='real')

    return {'CCimage_pix': None,
            'displacement_pix': None,
            'CCimage_subpix': subpix_correl,
            'displacement_subpix': subpix_dis,
            'CCimage': subpix_correl,
            'displacement': subpix_displacement}


def subpixelar_interp(subimg1, subimg2, num_points_interp=1,
                      method='RectBivariateSpline', kind='linear'):
    """
    Calculate the displacement field between subimg1 and subimg2.

    The first pass is a DIC with a pixel-size resolution.
    The deformed image is interpolated near the local displaced location
    and a second DIC is performed.

    Parameters
    ----------
    subimg1 : (N, N) ndarray
        Reference image.
    subimg2 : (M, M) ndarray
        Deformed image, M >= N.
    num_points_interp : int, optional
        Number of points introduced between fixed data points.
    method : str, optional
        Method used to interpolate. Must be `rbf` or `interp2d`.
    kind : str, optional
        Function used to interpolate. For valid parameters, see
        the `kind` argument of `np.interpolate.interp2d` or
        the `function` argument of `np.interpolate.Rbf`.

    Returns
    -------
    result : dict
        The keys of the dictionnary are:
            * 'CCimage_pix': Image of the pixellar cross-correlation
            * 'displacement_pix': Pixelar displacement
            * 'CCimage_subpix': Image of the subpixellar cross-correlation
            * 'displacement_subpix': Subpixelar displacement
            * 'CCimage': Cross-correlation at the best refinement
            * 'displacement': Cross-correlation at the best refinement

    Notes
    -----
    The Digital Image Correlation (DIC) uses a ZNCC correlator provided by
    `skimage.feature.match_template`.

    References
    ----------
    .. [1] Briechle and Hanebeck, "Template Matching using Fast Normalized
           Cross Correlation", Proceedings of the SPIE (2001).
           DOI:10.1117/12.421129
    .. [2] J. P. Lewis, "Fast Normalized Cross-Correlation", Industrial Light
           and Magic.

    Examples
    --------
    >>> from skimage import transform as tf
    >>> img1 = np.random.normal(size=(200, 200))
    >>> tform = tf.AffineTransform(scale=(1, 1), rotation=0,
                                   translation=(3, 2), shear=0.03)
    >>> img2 = tf.warp(img1, tform)
    >>> coord = 50, 50
    >>> subimg1, subimg2 = extract_subimg(img1, img2, coord)
    >>> res = subpixelar_interp(subimg1, subimg2)

    """
    #warnings.filterwarnings('error', '.*No more knots can be added because the number of B-spline.*')
    #warnings.filterwarnings('error', '.*A theoretically impossible result when finding a smoothing spline.*')

    # Check images are squares
    for si in (subimg1, subimg2):
        if si.shape[0] != si.shape[1]:
            raise ValueError('Images must have a dimension (N, N).')
    # Check that dimensions of img2 larger than img1
    if subimg2.shape[0] < subimg1.shape[0]:
            raise ValueError('subimg2 must have a dimension equal or larger than subimg1.')

    # 1st correlation
    pix_correl = match_template(subimg2, subimg1, pad_input=True)
    pix_peak = peak_local_max(pix_correl, num_peaks=1, indices=True)[0]
    pix_displacement = np.array(pix_peak - np.array(pix_correl.shape) / 2.,
                                dtype=np.int).tolist()
    dis_i, dis_j = pix_displacement

    # Prepare the 2nd correlation
    # Recover seek and bsize
    bsize = subimg1.shape[0]
    seek = int((subimg2.shape[0] - bsize) / 2)

    if seek + dis_i < 0 or seek + dis_j < 0 or seek+dis_i+bsize > subimg2.shape[0]\
       or seek+dis_j+bsize > subimg2.shape[1]:
        # It's too heavy to handle this case, so we don't seek for an optimized centered window
        subimg4 = subimg2.copy()
        peak_centered = False
    else:
        # subimg2 at the previous displacement to get smaller image, better perf
        subimg4 = subimg2[seek+dis_i:seek+dis_i+bsize, seek+dis_j:seek+dis_j+bsize]
        peak_centered = True

    # Interpolate
    subimg1_interp = interpolate_image(subimg1,
                                       num_points=num_points_interp,
                                       method=method,
                                       kind=kind)
    subimg4_interp = interpolate_image(subimg4,
                                       num_points=num_points_interp,
                                       method=method,
                                       kind=kind)
    # 2nd correlation
    subpix_correl = match_template(subimg4_interp, subimg1_interp,
                                   pad_input=True)

    subpix_peak = peak_prominent(subpix_correl)

    # Add the 2nd displacement
    correction = (subpix_peak - np.array(subpix_correl.shape) / 2.) / num_points_interp
    # Ensure that the correction is small, i.e. we detected the right peak
    if np.all(np.abs(correction) < 1):
        dis_i += correction[0]
        dis_j += correction[1]
    else:
        warn('Correction to the displacement after interpolation not small: %s' % correction)

    subpix_displacement = dis_i, dis_j

    return {'CCimage_pix': pix_correl,
            'displacement_pix': pix_displacement,
            'CCimage_subpix': subpix_correl,
            'displacement_subpix': subpix_displacement,
            'CCimage': subpix_correl,
            'displacement': subpix_displacement}


def subpixelar_interp_fit(subimg1, subimg2, num_points_interp=1,
                          method='RectBivariateSpline', kind='linear',
                          fit_method='biparabolic', fit_algorithm='leastsq'):
    """
    Calculate the displacement field between subimg1 and subimg2.

    The first pass is a DIC with a pixel-size resolution.
    The deformed image is interpolated near the local displaced location
    and a second DIC is performed.

    Parameters
    ----------
    subimg1 : (N, N) ndarray
        Reference image.
    subimg2 : (M, M) ndarray
        Deformed image, M >= N.
    num_points_interp : int, optional
        Number of points introduced between fixed data points.
    method : str, optional
        Method used to interpolate. Must be `rbf` or `interp2d`.
    kind : str, optional
        Function used to interpolate. For valid parameters, see
        the `kind` argument of `np.interpolate.interp2d` or
        the `function` argument of `np.interpolate.Rbf`.
    fit_method : str, optional
        Method used for the fit. #TODO
    fit_algorithm : str, optional
        Algorithm used for the fit. #TODO

    Returns
    -------
    result : dict
        The keys of the dictionnary are:
            * 'CCimage_pix': Image of the pixellar cross-correlation
            * 'displacement_pix': Pixelar displacement
            * 'CCimage_subpix': Image of the subpixellar cross-correlation
            * 'displacement_subpix': Subpixelar displacement
            * 'CCimage': Cross-correlation at the best refinement
            * 'displacement': Cross-correlation at the best refinement

    Notes
    -----
    The Digital Image Correlation (DIC) uses a ZNCC correlator provided by
    `skimage.feature.match_template`.

    References
    ----------
    .. [1] Briechle and Hanebeck, "Template Matching using Fast Normalized
           Cross Correlation", Proceedings of the SPIE (2001).
           DOI:10.1117/12.421129
    .. [2] J. P. Lewis, "Fast Normalized Cross-Correlation", Industrial Light
           and Magic.

    Examples
    --------
    >>> from skimage import transform as tf
    >>> img1 = np.random.normal(size=(200, 200))
    >>> tform = tf.AffineTransform(scale=(1, 1), rotation=0,
                                   translation=(3, 2), shear=0.03)
    >>> img2 = tf.warp(img1, tform)
    >>> coord = 50, 50
    >>> subimg1, subimg2 = extract_subimg(img1, img2, coord)
    >>> res = subpixelar_interp_fit(subimg1, subimg2)

    """
    #warnings.filterwarnings('error', '.*No more knots can be added because the number of B-spline.*')
    #warnings.filterwarnings('error', '.*A theoretically impossible result when finding a smoothing spline.*')

    # Check images are squares
    for si in (subimg1, subimg2):
        if si.shape[0] != si.shape[1]:
            raise ValueError('Images must have a dimension (N, N).')
    # Check that dimensions of img2 larger than img1
    if subimg2.shape[0] < subimg1.shape[0]:
            raise ValueError('subimg2 must have a dimension equal or larger than subimg1.')

    # 1st correlation
    pix_correl = match_template(subimg2, subimg1, pad_input=True)
    pix_peak = peak_local_max(pix_correl, num_peaks=1, indices=True)[0]
    pix_displacement = np.array(pix_peak - np.array(pix_correl.shape) / 2., dtype=np.int).tolist()
    dis_i, dis_j = pix_displacement

    # Prepare the 2nd correlation
    # Recover seek and bsize
    bsize = subimg1.shape[0]
    seek = int((subimg2.shape[0] - bsize) / 2)

    if seek + dis_i < 0 or seek + dis_j < 0 or seek+dis_i+bsize > subimg2.shape[0]\
       or seek+dis_j+bsize > subimg2.shape[1]:
        # It's too heavy to handle this case, so we don't seek for an optimized centered window
        subimg4 = subimg2.copy()
        peak_centered = False
    else:
        # subimg2 at the previous displacement to get smaller image, better perf
        subimg4 = subimg2[seek+dis_i:seek+dis_i+bsize, seek+dis_j:seek+dis_j+bsize]
        peak_centered = True

    # Interpolate
    subimg1_interp = interpolate_image(subimg1,
                                       num_points=num_points_interp,
                                       method=method,
                                       kind=kind)
    subimg4_interp = interpolate_image(subimg4,
                                       num_points=num_points_interp,
                                       method=method,
                                       kind=kind)
    # 2nd correlation
    subpix_correl = match_template(subimg4_interp, subimg1_interp,
                                   pad_input=True)

    if peak_centered:
        # The peak is necessarely near the center (small displacement)
        subpix_peak = peak_near_center_fitted(subpix_correl, halfwidth=num_points_interp,
                                              method=fit_method,
                                              algorithm=fit_algorithm)
    else:
        subpix_peak = peak_near_center_fitted(subpix_correl, halfwidth=None,
                                              method=fit_method,
                                              algorithm=fit_algorithm)

    # Add the 2nd displacement
    correction = (subpix_peak - np.array(subpix_correl.shape) / 2.) / num_points_interp
    # Ensure that the correction is small, i.e. we detected the right peak
    if np.all(np.abs(correction) < 1):
        dis_i += correction[1] # swapped!
        dis_j += correction[0]
    else:
        warn('Correction to the displacement after interpolation not small: %s' % correction)

    subpix_displacement = dis_i, dis_j

    return {'CCimage_pix': pix_correl,
            'displacement_pix': pix_displacement,
            'CCimage_subpix': subpix_correl,
            'displacement_subpix': subpix_displacement,
            'CCimage': subpix_correl,
            'displacement': subpix_displacement}
