#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import itertools
import warnings

try:
    from dask import compute, delayed
except ImportError:
    dask_available = False
else:
    dask_available = True

from .utils import pairwise


def pavement(img_shape, bsize=16, seek=0, overlap=0):
    """
    Calculate box positions to pave an image.

    The box size corresponds to the side number of pixel of the squared box.
    This method takes into account the overlap between the different
    subdmains as well as the distance over which the displace is seek.

    Parameters
    ----------
    img_shape : tuple
        Image shape.
    bsize : int, optional
        Box size.
    seek : int, optional
        Number of pixels considered to seek the displacement.
    overlap : int, optional
        Number of overlapping pixels on between two
        neighbouring boxes.

    Returns
    -------
    positions : tuple of two lists.
    """
    img_i_max, img_j_max = img_shape
    # The pavement should go from seek to
    # (img_size - seek - bsize)
    step = bsize - overlap
    ii = range(seek, img_i_max - seek - bsize + 1, step)
    jj = range(seek, img_j_max - seek - bsize + 1, step)
    return ii, jj


def gen_subset(img1, img2, bsize, seek, overlap):
    """
    Generator of subset images.

    Parameters
    ----------
    img1 : (N, M) ndarray
        Reference image.
    img2 : (N, M) ndarray
        Deformed image.
    bsize : int, optional
        Box size.
    seek : int, optional
        Number of pixels considered to seek the displacement.
    overlap : int, optional
        Number of overlapping pixels on between two
        neighbouring boxes.

    Returns
    -------
    i, j, subimg1, subimg2 : tuple.
        i, j are the corner positions from `pavement`.
    """
    ii, jj = pavement(img1.shape, bsize=bsize, overlap=overlap, seek=seek)
    for i, j in itertools.product(ii, jj):
        subimg1 = img1[i:i+bsize, j:j+bsize]
        subimg2 = img2[i-seek:i+bsize+seek, j-seek:j+bsize+seek]
        yield i, j, subimg1, subimg2


def _dic_task(i, j, subimg1, subimg2, bsize, subset_dic, stats, **kwargs):
    """
    Elementary DIC task that returns positions and displacements.
    """
    res = subset_dic(subimg1, subimg2, **kwargs)

    dis_i, dis_j = res['displacement']

    pos_i = (i + int(bsize / 2.))
    pos_j = (j + int(bsize / 2.))

    if stats:
        return (dis_i, dis_j, pos_i, pos_j,
                res['CCimage'].mean(), res['CCimage'].std(),
                res['CCimage'].max(), res['CCimage'].min())
    else:
        return dis_i, dis_j, pos_i, pos_j


def dic(img1, img2, subset_dic, bsize=16, seek=None, overlap=None,
        stats=False, parallel=False, **kwargs):
    """
    Calculate the displacement field between img1 and img2.

    The algorithm to perform the DIC is specified, at the subset level,
    by a callable `subset_dic`.

    Parameters
    ----------
    img1 : (N, M) ndarray
        Reference image.
    img2 : (N, M) ndarray
        Deformed image.
    subset_dic : callable
        subset function to perform DIC.
    bsize : int, optional
        Box size.
    seek : int, optional
        Number of pixels considered to seek the displacement.
        If `None`, the value is set to `bsize`.
    overlap : int, optional
        Number of overlapping pixels on between two
        neighbouring boxes. If `None`, the value is set to
        bsize / 2.
    stats : bool, optional
        If True, add statistics about the peak correlation
        in the output.
    parallel : bool, optional
        If True, use dask (if available) to use multiprocessing.
    kwargs :
        Keyword arguments to pass to `subset_dic`.

    Returns
    -------
    displacements : pd.DataFrame
        Positions (y, x) of box centers and displacements (dy, dx).

    Examples
    --------
    >>> from skimage import transform as tf
    >>> from skDIC.subsetdic import pixelar
    >>> img1 = np.random.normal(size=(200, 200))
    >>> tform = tf.AffineTransform(scale=(1, 1), rotation=0,
                                   translation=(3, 2), shear=0.03)
    >>> img2 = tf.warp(img1, tform)
    >>> displacements = dic(img1, img2, pixelar)

    """
    # Number of pixels considered to seek the displacement.
    if seek is None:
        seek = bsize
    # Number of overlapping pixels
    if overlap is None:
        overlap = int(bsize / 2.)

    if dask_available and parallel:
        values = [delayed(_dic_task)(i, j, s1, s2, bsize, subset_dic, stats, **kwargs)
                  for i, j, s1, s2 in gen_subset(img1, img2, bsize, seek, overlap)]
        data = compute(*values, scheduler='processes')
    else:
        data = [_dic_task(i, j, s1, s2, bsize, subset_dic, stats, **kwargs)
                for i, j, s1, s2 in gen_subset(img1, img2, bsize, seek, overlap)]

    data = np.column_stack(np.array(data))
    data_dict = {'dy': data[0],
                 'dx': data[1],
                 'y': data[2],
                 'x': data[3],
                 }
    if stats:
        data_dict['pix_cor_mean'] = data[4]
        data_dict['pix_cor_std'] = data[5]
        data_dict['pix_cor_max'] = data[6]
        data_dict['pix_cor_min'] = data[7]

    displacement = pd.DataFrame(data_dict)
    # Conpute the norm of the displacement.
    displacement['dl'] = np.sqrt(displacement.dx**2 + displacement.dy**2)
    return displacement


def batch_dic(frames, subset_dic, bsize=16, seek=None, overlap=None,
              stats=False, parallel_dic=False, **kwargs):
    """
    Calculate the displacement field between consecutive images.

    The algorithm to perform the DIC is specified, at the subset level,
    by a callable `subset_dic`.

    Parameters
    ----------
    frames : list (or iterable) of images
        A series of images.
    subset_dic : callable
        subset function to perform DIC.
    bsize : int, optional
        Box size.
    seek : int, optional
        Number of pixels considered to seek the displacement.
        If `None`, the value is set to `bsize`.
    overlap : int, optional
        Number of overlapping pixels on between two
        neighbouring boxes. If `None`, the value is set to
        bsize / 2.
    stats : bool, optional
        If True, add statistics about the peak correlation
        in the output.
    parallel_dic : bool, optional
        If True, set `parallel=True` to `dic()`. Then, Dask
        is used to perform multiprocessing at the level of
        the image processing.
    kwargs :
        Keyword arguments to pass to `subset_dic`.

    Returns
    -------
    displacements : pd.DataFrame
        Positions (y, x) of box centers and displacements (dy, dx).
        A column frame contains the frame number.

    """
    def task(i, img1, img2, subset_dic, bsize, seek, overlap, stats, **kwargs):
        """
        Task that performs DIC betweent two images and add the frame number to the result.
        """
        dis = dic(img1, img2, subset_dic,
                  bsize=bsize,
                  seek=seek,
                  overlap=overlap,
                  stats=stats,
                  **kwargs)
        dis['frame'] = i
        return dis

    if dask_available:
        values = [delayed(task)(i, img1, img2, subset_dic, bsize, seek, overlap, stats, parallel=parallel_dic, **kwargs)
                  for i, (img1, img2) in pairwise(frames)]
        all_dis = compute(*values, scheduler='processes')
    else:
        all_dis = [task(i, img1, img2, subset_dic, bsize, seek, overlap, stats, **kwargs)
                   for i, (img1, img2) in pairwise(frames)]

    return pd.concat(all_dis).reset_index(drop=True)


def compute_cumulative_displacements(df):
    """
    Compute the cumulative displacements of a dataframe.

    :param df: dataframe

    """

    all_frames = df.frame.unique()
    all_frames.sort()

    # initialize the new data with trivial first case
    new_data =  df[df.frame == all_frames[0]].copy()
    new_data['cumdx'] = new_data.dx
    new_data['cumdy'] = new_data.dy

    for frame in all_frames[1:]:
        f1 = df[df.frame == frame-1]
        f2 = df[df.frame == frame]
        f1 = f1.sort_values(by=['x', 'y'], ascending=[True, True])
        f1 = f1.reset_index()
        f2 = f2.sort_values(by=['x', 'y'], ascending=[True, True])
        f2 = f2.reset_index()
        f2['cumdx'] = f1.dx + f2.dx
        f2['cumdy'] = f1.dy + f2.dy
        f2['cumdl'] = np.sqrt(f2.cumdx**2 + f2.cumdy**2)
        new_data = pd.concat((new_data, f2))

    # Sort, reset pandas indices, and cleanup
    new_data.sort_values(by=['frame', 'x', 'y'], ascending=[True, True, True], inplace=True)
    new_data.reset_index(inplace=True, drop=True)
    new_data.drop(columns='index', inplace=True)
    return new_data
