#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal, assert_array_less, assert_raises
from scipy.ndimage.interpolation import shift
from skimage import transform as tf
from skimage.feature.peak import peak_local_max

from skDIC.dic import pavement, dic
from skDIC.subsetdic import pixelar, subpixelar_interp
from skDIC.testing import assert_almost_identical


class TestPavement(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_position_zero_seek_zero_overlap(self):
        shape = (20, 50)
        bsize = 10
        res1, res2 = pavement(shape, bsize=bsize, seek=0, overlap=0)
        assert_equal(res1, np.array((0, 10)))
        assert_equal(res2, np.array((0, 10, 20, 30, 40)))

    def test_position_zero_seek_overlap(self):
        shape = (20, 50)
        bsize = 10
        res1, res2 = pavement(shape, bsize=bsize, seek=0, overlap=5)
        assert_equal(res1, np.array((0, 5, 10)))
        assert_equal(res2, np.array((0, 5, 10, 15, 20, 25, 30, 35, 40)))

    def test_position_seek_zero_overlap(self):
        shape = (30, 60)
        bsize = 10
        seek = 5
        res1, res2 = pavement(shape, bsize=bsize, seek=seek, overlap=0)
        assert_equal(res1, np.array((5, 15)))
        assert_equal(res2, np.array((5, 15, 25, 35, 45)))

    def test_position_nonproportional_imgshape(self):
        shape = (54, 34)
        res1, res2 = pavement(shape, bsize=8, seek=8, overlap=4)
        # Should stop at 54 - 2*8 = 38
        assert_equal(res1, np.array((8, 12, 16, 20, 24, 28, 32, 36)))

    def test_position_nonproportional_imgshape2(self):
        shape = (56, 34)
        res1, res2 = pavement(shape, bsize=8, seek=8, overlap=4)
        # Should stop at 56 - 2*8 = 40
        assert_equal(res1, np.array((8, 12, 16, 20, 24, 28, 32, 36, 40)))


class TestDICpixel(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

    def test_displacement_int(self):
        img = np.random.normal(size=(32, 32))
        img2 = shift(img, (1, 2))
        displacement = dic(img, img2, pixelar, bsize=8)
        assert_equal(displacement.dy, np.ones(len(displacement.dy)) * 1)
        assert_equal(displacement.dx, np.ones(len(displacement.dy)) * 2)

    def test_displacement_int_stats(self):
        # Here, we only check that the use of stats
        # does not through an exception.
        img = np.random.normal(size=(32, 32))
        img2 = shift(img, (1, 2))
        displacement = dic(img, img2, pixelar, bsize=8, stats=True)
        assert_equal(displacement.dy, np.ones(len(displacement.dy)) * 1)
        assert_equal(displacement.dx, np.ones(len(displacement.dy)) * 2)


class TestDICsubpixel(unittest.TestCase):
    def setUp(self):
        np.random.seed(3)
        self.img = np.random.normal(size=(32, 32))

    def test_RectBivariateSpline_displacement_int(self):
        method = 'RectBivariateSpline'
        img2 = shift(self.img, (1, 2))
        displacement = dic(self.img, img2, subpixelar_interp, bsize=8, method=method)
        assert_array_almost_equal(displacement.dy, np.ones(len(displacement.dy)) * 1)
        assert_array_almost_equal(displacement.dx, np.ones(len(displacement.dx)) * 2)

    def test_rbf_displacement_int(self):
        method = 'rbf'
        img2 = shift(self.img, (1, 2))
        displacement = dic(self.img, img2, subpixelar_interp, bsize=8, method=method)
        assert_array_almost_equal(displacement.dy, np.ones(len(displacement.dy)) * 1)
        assert_array_almost_equal(displacement.dx, np.ones(len(displacement.dx)) * 2)

    def test_interp2d_displacement_int(self):
        method = 'interp2d'
        img2 = shift(self.img, (1, 2))
        displacement = dic(self.img, img2, subpixelar_interp, bsize=8, method=method)
        assert_array_almost_equal(displacement.dy, np.ones(len(displacement.dy)) * 1)
        assert_array_almost_equal(displacement.dx, np.ones(len(displacement.dx)) * 2)

    def test_RectBivariateSpline_displacement_float(self):
        method = 'RectBivariateSpline'
        shift = 1.3
        npoints = 5
        tform = tf.AffineTransform(scale=(1, 1), rotation=0, translation=(-shift, 0), shear=0)
        img2 = tf.warp(self.img, tform, order=3)
        displacement = dic(self.img, img2, subpixelar_interp, bsize=8, num_points_interp=npoints, method=method)

        assert_array_almost_equal(displacement.dy, np.zeros(len(displacement.dy)))
        expected = np.ones(len(displacement.dx)) * shift
        assert_array_less(np.abs(displacement.dx - expected[1]), np.ones(expected[1].shape) * 0.31)

    def test_rbf_displacement_float(self):
        method = 'rbf'
        shift = 1.3
        npoints = 5
        tform = tf.AffineTransform(scale=(1, 1), rotation=0, translation=(-shift, 0), shear=0)
        img2 = tf.warp(self.img, tform, order=3)
        displacement = dic(self.img, img2, subpixelar_interp, bsize=8, num_points_interp=npoints, method=method)

        assert_array_almost_equal(displacement.dy, np.zeros(len(displacement.dy)))
        expected = np.ones(len(displacement.dx)) * shift
        assert_array_less(np.abs(displacement.dx - expected[1]), np.ones(expected[1].shape) * 0.31)

    @unittest.skip('really off...')
    def test_interp2d_displacement_float(self):
        method = 'interp2d'
        shift = 1.3
        npoints = 5
        tform = tf.AffineTransform(scale=(1, 1), rotation=0, translation=(-shift, 0), shear=0)
        img2 = tf.warp(self.img, tform, order=3)
        displacement = dic(self.img, img2, subpixelar_interp, bsize=8, num_points_interp=npoints, method=method)

        assert_almost_identical(displacement.dy, np.zeros(len(displacement.dy)))
        expected = np.ones(len(displacement.dx)) * shift
        assert_array_less(np.abs(displacement.dx - expected[1]), np.ones(expected[1].shape) * 0.31)

    def test_RectBivariateSpline_displacement_float2(self):
        method = 'RectBivariateSpline'
        shift = 1.7
        npoints = 15
        tform = tf.AffineTransform(scale=(1, 1), rotation=0, translation=(-shift, 0), shear=0)
        img2 = tf.warp(self.img, tform, order=3)
        displacement = dic(self.img, img2, subpixelar_interp, bsize=8, num_points_interp=npoints, method=method)

        assert_array_almost_equal(displacement.dy, np.zeros(len(displacement.dy)))
        expected = np.ones(len(displacement.dx)) * shift
        assert_array_less(np.abs(displacement.dx - expected[1]), np.ones(expected[1].shape) * 0.31)

    def test_rbf_displacement_float2(self):
        method = 'rbf'
        shift = 1.7
        npoints = 15
        tform = tf.AffineTransform(scale=(1, 1), rotation=0, translation=(-shift, 0), shear=0)
        img2 = tf.warp(self.img, tform, order=3)
        displacement = dic(self.img, img2, subpixelar_interp, bsize=8, num_points_interp=npoints, method=method)

        assert_array_almost_equal(displacement.dy, np.zeros(len(displacement.dy)))
        expected = np.ones(len(displacement.dx)) * shift
        assert_array_less(np.abs(displacement.dx - expected[1]), np.ones(expected[1].shape) * 0.31)

    @unittest.skip('really off...')
    def test_interp2d_displacement_float2(self):
        method = 'interp2d'
        shift = 1.7
        npoints = 15
        tform = tf.AffineTransform(scale=(1, 1), rotation=0, translation=(-shift, 0), shear=0)
        img2 = tf.warp(self.img, tform, order=3)
        displacement = dic(self.img, img2, subpixelar_interp, num_points_interp=npoints, method=method)

        assert_almost_identical(displacement.dy, np.zeros(len(displacement.dy)), tol=0.12)
        expected = np.ones(len(displacement.dx)) * shift
        assert_array_less(np.abs(displacement.dx - expected[1]), np.ones(expected[1].shape) * 0.31)


if __name__ == '__main__':
    unittest.main()
