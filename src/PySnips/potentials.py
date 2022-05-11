# -*- Mode: python3; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
"""Potentials."""

def LJ(z, sigma, epsilon):
    """Returns the 12,6 LJ potential."""
    sigz6 = (sigma / z) ** 6
    return 4 * epsilon * (sigz6 ** 2 - sigz6)
