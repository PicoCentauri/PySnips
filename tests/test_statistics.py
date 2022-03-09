# -*- Mode: python3; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
"""Test statistics module."""
import numpy as np

from PySnips import statistics


def test_rmse():
    """Test RMSE."""
    ypred = np.arange(10)
    y = np.arange(10)
    assert statistics.get_rmse(ypred, y) == 0
