#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import unittest
from numpy.testing import (assert_array_almost_equal,
                           assert_equal, assert_array_less)
from skimage.feature import peak_local_max
from skDIC.peak import peak_prominent, peak_near_center, peak_near_center_fitted


class TestPeakProminent(unittest.TestCase):

    def test_peak_prominent(self):
        img = np.zeros((100, 200))
        # Main peak centered around 50, 100
        img[50:51, 100:101] = 3
        img[20:21, 140:141] = 1
        img[80:81, 180:181] = 1
        expected = (50, 100)
        width = 10
        result = peak_near_center(img, width)
        assert_equal(result, expected)


class TestPeakNearCenter(unittest.TestCase):

    def test_peak_near_center(self):
        img = np.zeros((100, 200))
        # Peak centered around 50, 100
        img[49:52, 99:102] = 2
        img[50:51, 100:101] = 3
        expected = peak_local_max(img)[0]
        width = 10
        result = peak_near_center(img, width)
        assert_equal(result, expected)

    def test_peak_near_center_large_width(self):
        img = np.zeros((100, 200))
        # Peak centered around 50, 100
        img[49:52, 99:102] = 2
        img[50:51, 100:101] = 3
        expected = peak_local_max(img)[0]
        width = 300
        result = peak_near_center(img, width)
        assert_equal(result, expected)

    def test_peak_near_center_with_secondary_peak(self):
        img = np.zeros((100, 200))
        # Peak centered around 50, 100
        img[50:51, 100:101] = 3
        img[79:82, 39:42] = 20
        expected = peak_local_max(img)[0]
        width = 30
        result = peak_near_center(img, width)
        assert_equal(result, expected)


class TestPeakNearCenterFitted(unittest.TestCase):

    def test_peak_near_center(self):
        img = np.zeros((100, 200))
        # Peak centered around 50, 100
        img[49:52, 99:102] = 2
        img[50:51, 100:101] = 3
        expected = peak_local_max(img)[0]
        width = 10
        result = peak_near_center_fitted(img, width)
        assert_array_almost_equal(result, expected)

    def test_peak_near_center_with_secondary_peak(self):
        img = np.zeros((100, 200))
        # Peak centered around 50, 100
        img[50:51, 100:101] = 3
        img[79:82, 39:42] = 20
        expected = peak_local_max(img)[0]
        width = 10
        result = peak_near_center_fitted(img, width)
        assert_array_almost_equal(result, expected)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
