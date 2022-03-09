# -*- Mode: python3; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
"""Definitions for plotting with matplotlib."""

import matplotlib.pyplot as plt
import mpltex
import numpy as np


TAB10 = plt.rcParams["axes.prop_cycle"].by_key()["color"]
FSIZE = np.array([mpltex.acs._width, mpltex.acs._height])
