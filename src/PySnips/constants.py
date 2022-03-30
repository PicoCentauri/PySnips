# -*- Mode: python3; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
"""Constants and conversions not defined elsewhere."""

import scipy.constants

# Volume of a SPC/E water molecule nm^3
v_spce = 0.0304

# Energy conversions
_kNa = scipy.constants.kilo / scipy.constants.Avogadro
eV_to_KJmol = scipy.constants.electron_volt / _kNa
KbT_to_KJmol = 300 * scipy.constants.Boltzmann / _kNa

# Back conversions
kJmol_to_eV = 1 / eV_to_KJmol
KJmol_to_KbT = 1 / KbT_to_KJmol
