# -*- Mode: python3; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
"""Functions for reading and writen certain files."""

from MDAnalysis.lib.mdamath import triclinic_box


def write_xyz(fname, positions, chemical_symbols, cell):
    """Writes coordinate of N atoms to input files for i-PI simulation.

    Paramaters
    ----------
    fname : str
        name of the outfile
    positions : np.ndarray
        positions with shape (N, 3)
    chemical_symbols : list
        liist of chemical symbols of len N
    cell : np.ndarray
        cell vectors with shape (3,3)
    """
    if positions.shape[0] != len(chemical_symbols):
        raise ValueError(
            f"Number of particles ({positions.shape[0]}) in "
            f"positions and chemical_symbols "
            f"({len(chemical_symbols)}) does not agree."
        )

    _write_xyz_rascal(fname, positions, chemical_symbols, cell)
    _write_xyz_ipi(fname, positions, chemical_symbols, cell)


def _positions_to_str(positions, chemical_symbols):
    out = ""

    for a, pos in zip(chemical_symbols, positions):
        f_pos = [f"{p:>12.5e}" for p in pos]
        out += f"{a:>8} {' '.join(f_pos)}\n"

    return out


def _write_xyz_rascal(fname, positions, chemical_symbols, cell):
    out = f"{len(positions)}\n"
    out += f"Lattice='{' '.join(map(str, cell.flatten()))}'\n"
    out += _positions_to_str(positions, chemical_symbols)

    with open(f"{fname}.extxyz", "w") as f:
        f.write(out)


def _write_xyz_ipi(fname, positions, chemical_symbols, cell):
    out = f"{len(positions)}\n"
    f_cellpar = [f"{i:11.5f}" for i in triclinic_box(cell)]

    out += f"# CELL(abcABC):{' '.join(f_cellpar)}  "
    out += "cell{angstrom}  Traj: positions{angstrom} "
    out += f"Step: {0:>11d}  Bead: {10:7d}\n"
    out += _positions_to_str(positions, chemical_symbols)

    with open(f"{fname}.xyz", "w") as f:
        f.write(out)
