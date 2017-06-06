#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal, assert_array_less
from scipy.ndimage.interpolation import shift
from skimage import transform as tf
from skimage.feature.peak import peak_local_max

from skDIC.subsetdic import pixelar, subpixelar_dft, subpixelar_interp, subpixelar_interp_fit
from skDIC.testing import assert_almost_identical


class TestsubsetDIC(unittest.TestCase):
    def setUp(self):
        np.random.seed(3)
        self.shift_int = (1, 2)
        self.shift_float = (1.71, 2.18)
        i, j = 15, 15
        bsize = 8
        seek = 4
        self.img1 = np.random.normal(size=(32, 32))
        self.img2 = shift(self.img1, self.shift_int)
        self.img3 = shift(self.img1, self.shift_float)
        self.subimg1 = self.img1[i:i+bsize, j:j+bsize]
        self.subimg2 = self.img2[i-seek:i+bsize+seek, j-seek:j+bsize+seek]
        self.subimg2_samesize = self.img2[i:i+bsize, j:j+bsize]
        self.subimg3 = self.img3[i-seek:i+bsize+seek, j-seek:j+bsize+seek]

    def test_rectangular_images(self):
        i = 15
        j = 15
        bsize = 8
        seek = 4
        subimg1 = self.img1[i:i+bsize, j:j+bsize+1]
        subimg2 = self.img2[i-seek:i+bsize+seek, j-seek:j+bsize+seek]
        with pytest.raises(ValueError):
            pixelar(subimg1, subimg2)
        with pytest.raises(ValueError):
            subpixelar_interp(subimg1, subimg2)
        with pytest.raises(ValueError):
            subpixelar_interp_fit(subimg1, subimg2)

    # Pixelar

    def test_pixelar_no_displacement(self):
        res = pixelar(self.subimg1, self.subimg1, correlator='ZNCC')
        assert_equal(res['displacement_pix'], (0, 0))
        res = pixelar(self.subimg1, self.subimg1, correlator='DFT')
        assert_equal(res['displacement_pix'], (0, 0))

    def test_pixelar_displacement_int(self):
        res = pixelar(self.subimg1, self.subimg2, correlator='ZNCC')
        assert_equal(res['displacement_pix'], self.shift_int)
        res = pixelar(self.subimg1, self.subimg1, correlator='DFT')
        assert_equal(res['displacement_pix'], (0, 0))

    def test_pixelar_displacement_float(self):
        res = pixelar(self.subimg1, self.subimg3, correlator='ZNCC')
        assert_equal(res['displacement_pix'], np.round(self.shift_float))
        res = pixelar(self.subimg1, self.subimg3, correlator='DFT')
        assert_equal(res['displacement_pix'], np.round(self.shift_float))

    # Subpixelar_dft

    def test_SubpixelarInterpFit_no_displacement(self):
        res = subpixelar_dft(self.subimg1, self.subimg1)
        assert_array_almost_equal(res['displacement_subpix'], (0, 0), decimal=2)

    def test_SubpixelarInterpFit_displacement_int(self):
        res = subpixelar_dft(self.subimg1, self.subimg2)
        assert_array_almost_equal(res['displacement_subpix'], self.shift_int, decimal=2)

    def test_SubpixelarInterpFit_displacement_float(self):
        res = subpixelar_dft(self.subimg1, self.subimg3)
        assert_array_almost_equal(res['displacement_subpix'], self.shift_float, decimal=2)

    # Subpixelar_interp

    def test_SubpixelarInterp_same_size(self):
        assert_equal(self.subimg1.shape, self.subimg2_samesize.shape)
        res = subpixelar_interp(self.subimg1, self.subimg2_samesize)
        assert_equal(res['displacement_subpix'], self.shift_int)

    def test_SubpixelarInterp_no_displacement(self):
        res = subpixelar_interp(self.subimg1, self.subimg1)
        assert_equal(res['displacement_subpix'], (0, 0))

    def test_SubpixelarInterp_displacement_int(self):
        res = subpixelar_interp(self.subimg1, self.subimg2)
        assert_equal(res['displacement_subpix'], self.shift_int)

    def test_SubpixelarInterp_displacement_float(self):
        res = subpixelar_interp(self.subimg1, self.subimg3)
        assert_equal(res['displacement_subpix'], np.round(self.shift_float))

    # Subpixelar_interp_fit
    def test_SubpixelarInterpFit_same_size(self):
        assert_equal(self.subimg1.shape, self.subimg2_samesize.shape)
        res = subpixelar_interp_fit(self.subimg1, self.subimg2_samesize)
        assert_equal(res['displacement_subpix'], self.shift_int)

    def test_SubpixelarInterpFit_no_displacement(self):
        res = subpixelar_interp_fit(self.subimg1, self.subimg1)
        assert_array_almost_equal(res['displacement_subpix'], (0, 0), decimal=2)

    def test_SubpixelarInterpFit_displacement_int(self):
        res = subpixelar_interp_fit(self.subimg1, self.subimg2)
        assert_array_almost_equal(res['displacement_subpix'], self.shift_int, decimal=2)

    def test_SubpixelarInterpFit_displacement_float(self):
        res = subpixelar_interp_fit(self.subimg1, self.subimg3)
        assert_array_almost_equal(res['displacement_subpix'], self.shift_float, decimal=1)
