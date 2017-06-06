#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import unittest
from numpy.testing import (assert_array_almost_equal,
                           assert_equal, assert_array_less)
from skimage.feature import peak_local_max
from skDIC.peak import peak_near_center_fitted as subpixel_peak

algorithm = 'lmfit'

@unittest.skip('')
def test_centered_peak():
    img = np.zeros((40, 40))
    # Peak centered at 14,24
    img[12:17, 22:27] = 1
    img[13:16, 23:26] = 2
    img[14:15, 24:25] = 3
    expected = peak_local_max(img)
    assert_equal(expected[0], (14, 24))

#    result_gaussian = subpixel_peak(img, method='gaussian', algorithm=algorithm)
#    assert_array_almost_equal(result_gaussian, expected)
    result_quadrature = subpixel_peak(img, method='quadrature', algorithm=algorithm)
    assert_array_almost_equal(result_quadrature, expected)
    result_biparabolic = subpixel_peak(img, method='biparabolic', algorithm=algorithm)
    assert_array_almost_equal(result_biparabolic, expected)


@unittest.skip('Not implemented')
def test_offcentered_peak_gaussian():
    img = np.zeros((40, 40))
    # Peak centered at 14,24
    img[11:16, 21:26] = 1
    img[13:16, 23:26] = 2
    img[14:15, 24:25] = 4
    expected = peak_local_max(img, min_distance=3)[0]
    assert_equal(expected, (14, 24))

    #result_gaussian = subpixel_peak(img, num_peaks=1, method='gaussian', algorithm=algorithm)[0]
    #assert_array_less(result_gaussian, expected)
    #assert_array_less(expected - 2, result_gaussian)


def test_offcentered_peak_quadrature():
    img = np.zeros((40, 40))
    # Peak centered at 14,24
    img[11:16, 21:26] = 1
    img[13:16, 23:26] = 2
    img[14:15, 24:25] = 4
    expected = peak_local_max(img, min_distance=3)[0]
    assert_equal(expected, (14, 24))

    #result_quadrature = subpixel_peak(img, num_peaks=2, method='quadrature', algorithm=algorithm)[0]
    #assert_array_less(result_quadrature, expected + 1e-3) # + epsilon
    #assert_array_less(expected - 2, result_quadrature)


def test_offcentered_peak_biparabolic():
    img = np.zeros((40, 40))
    # Peak centered at 14,24
    img[11:16, 21:26] = 1
    img[13:16, 23:26] = 2
    img[14:15, 24:25] = 4
    expected = peak_local_max(img, min_distance=3)[0]
    assert_equal(expected, (14, 24))
    #result_biparabolic = subpixel_peak(img, num_peaks=1, method='biparabolic', algorithm=algorithm)[0]
    #assert_array_less(result_biparabolic, expected)
    #assert_array_less(expected - 2, result_biparabolic)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
