# -*- Mode: python3; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
"""Definitions for plotting with matplotlib."""

import matplotlib.pyplot as plt
import mpltex
import numpy as np


tab10 = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fsize = np.array([mpltex.acs._width, mpltex.acs._height])


def add_subplotlabels(fig, ax, labels, loc="upper left"):
    """Add a labels to each axis of a figure."""
    assert len(ax) == len(labels)

    offset = 4 / 72
    if loc == "upper left":
        xt = offset
        yt = -offset
        x = 0.0
        y = 1.0
        va = "top"
        ha = "left"
    elif loc == "lower left":
        xt = offset
        yt = offset
        x = 0.0
        y = 0.0
        va = "bottom"
        ha = "left"
    elif loc == "lower right":
        xt = -2 * offset
        yt = offset
        x = 1.0
        y = 0.0
        va = "bottom"
        ha = "right"
    elif loc == "upper right":
        xt = -2 * offset
        yt = -offset
        x = 1.0
        y = 1.0
        va = "top"
        ha = "right"
    else:
        raise ValueError(f"loc={loc} is not a valid location")

    bbox = dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3.0)
    for i, s in enumerate(labels):
        trans = plt.matplotlib.transforms.ScaledTranslation(xt, yt, fig.dpi_scale_trans)
        ax[i].text(x, y, s, transform=ax[i].transAxes + trans, va=va, ha=ha, bbox=bbox)
