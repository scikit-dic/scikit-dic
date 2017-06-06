#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .utils import interpolate_nan


def plot_subset_correlation(results):
    """
    Plot the correlation peak from subset DIC.

    Parameters
    ----------
    results : tuple
        output from an subset DIC function.
    """
    #peak = results[-2]
    dis_x, dis_y = results['displacement']
    try:
        correlation_img = results['CCimage_subpix']
    except KeyError:
        correlation_img = results['CCimage_pix']

    #plt.plot(peak[1], peak[0], 'ro')
    plt.imshow(correlation_img)
    plt.colorbar()
    plt.title(" Max: %f; Min: %f; Mean: %f; Std: %f; Displacement: %f, %f" % (correlation_img.max(),
                                                                              correlation_img.min(),
                                                                              correlation_img.mean(),
                                                                              correlation_img.std(),
                                                                              dis_x, dis_y))


def plot_displacement(img, displacements, every=10, scale=1):
    """
    Plot the displacement over the original image.

    Parameters
    ----------
    img : ndarray
        Background image.
    displacements : pd.DataFrame
        Positions (y, x) and displacements (dy, dx).
    every : int, optional
        Show one over `every` displacements.
    scale : float, optional
        Scale for arrows.

    """
    scale = displacements.dl.max() / 10. * scale

    # Make sure that data are correctly sorted.
    data = displacements.sort_values(by=['x', 'y'], ascending=[True, True])

    plt.quiver(data['x'][::every],
               data['y'][::every],
               data['dx'][::every],
               -data['dy'][::every],
               data['dl'][::every],
               pivot='middle', headwidth=5, headlength=5,
               units='x', scale=scale,
               cmap=plt.cm.viridis)

    plt.colorbar()
    plt.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    plt.axis('off')


def plot_magnitude(img, displacements):
    """
    Plot a map showing the magnitude of the displacements.

    Parameters
    ----------
    img : ndarray
        Background image.
    displacements : pd.DataFrame
        Positions (y, x) and displacements (dy, dx).

    """
    map_img = np.zeros(img.shape) * np.nan

    # Make sure that data are correctly sorted.
    data = displacements.sort_values(by=['x', 'y'], ascending=[True, True])

    for i, row in data.iterrows():
        map_img[int(row['y']), int(row['x'])] = row['dl']

    map_img = interpolate_nan(map_img)
    plt.imshow(map_img, cmap=plt.cm.viridis, interpolation='nearest')
    plt.colorbar()


def plot_boxes(img, displacements):
    """
    Plot the boxes used for correlation.

    Parameters
    ----------
    displacements : pd.DataFrame
        Positions (y, x) and displacements (dy, dx).
    every : int, optional
        Show one over `every` displacements.

    Notes
    -----
    This function is still under development.

    """
    bbox = 8

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, aspect='equal')
    ax2.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    colors = ('y', 'r')
    colors = ('y')
    for i, el in displacements.iterrows():
        pos = np.array((el['x'], el['y']))
        ax2.add_patch(
            patches.Rectangle(
                pos - int(bbox/2.),
                bbox,
                bbox,
                color=colors[int(i+1) % len(colors)],
                fill=True,
                alpha=.5,
            )
        )
