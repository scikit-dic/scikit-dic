#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal, assert_array_less
from skimage._shared._warnings import expected_warnings

from skDIC.utils import interpolate_image, entropy


class TestEntropy(unittest.TestCase):
    def test_uniform_image(self):
        img = np.zeros((10, 10))
        with expected_warnings(['invalid value encountered in true_divide']):
            res = entropy(img)
        assert_equal(res, np.nan)


class TestInterpolateImage(unittest.TestCase):
    def test_zero_point_rbf(self):
        img = np.zeros((6, 9))
        npoints = 0
        res = interpolate_image(img, num_points=npoints, method='rbf', kind='cubic')
        expected = np.zeros((6 * (npoints + 1), 9 * (npoints + 1)))
        assert_equal(res, expected)

    def test_flat_image_rbf(self):
        img = np.zeros((6, 9))
        npoints = 4
        res = interpolate_image(img, num_points=npoints, method='rbf', kind='cubic')
        expected = np.zeros((6 * (npoints + 1), 9 * (npoints + 1)))
        assert_equal(res, expected)

    def test_zero_point_interp2d(self):
        img = np.zeros((6, 9))
        npoints = 0
        res = interpolate_image(img, num_points=npoints, method='interp2d', kind='cubic')
        expected = np.zeros((6 * (npoints + 1), 9 * (npoints + 1)))
        assert_equal(res, expected)

    def test_flat_image_interp2d(self):
        img = np.zeros((6, 9))
        npoints = 4
        res = interpolate_image(img, num_points=npoints, method='interp2d', kind='cubic')
        expected = np.zeros((6 * (npoints + 1), 9 * (npoints + 1)))
        assert_equal(res, expected)
