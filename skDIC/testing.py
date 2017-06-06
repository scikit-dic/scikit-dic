#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing.utils import build_err_msg


def assert_almost_identical(x, y, tol=1e-2):
    """

    """
    unequal_pos = np.where(x != y)
    error = len(unequal_pos[0]) / x.size
    try:
        assert(error < tol)
    except AssertionError:
        msg = build_err_msg([error, tol], 'Not almost identical')
        raise AssertionError(msg)
