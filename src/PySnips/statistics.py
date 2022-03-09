# -*- Mode: python3; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
"""Functions for statistical calculations."""

import numpy as np


def get_rmse(ypred, y):
    """Root mean square deviayion betweeen `ypred` and `y`."""
    return np.sqrt(np.mean((ypred - y)**2))
