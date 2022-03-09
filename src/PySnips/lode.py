# -*- Mode: python3; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
"""Functions for working with the Long Distance Equivariants (LODE)."""

import numpy as np


def _get_feat_vec_dict(frame_feature, frame, l_species):
    """Extract features from pylode.

    Parameters
    ----------
    frame_feature : np.array
        features of shape (n_atoms, n_elements, max_radial, max_angular)
    frame : ase.atoms.Atoms
        Atoms of a trajectory
    l_species : list
        Element names corresponding to the elements in the second index of
        `frame_feature`.

    Returns
    -------
    feat_vec_dict : dict
        dictionary containing all possible element pair tuples as keys and the
        feature values
    """
    r_species = frame.get_chemical_symbols()

    feature_shape = frame_feature.shape
    n_atoms = feature_shape[0]
    n_elements = feature_shape[1]
    max_radial = feature_shape[2]
    max_angular = feature_shape[3]

    if n_elements != len(l_species):
        raise ValueError(f"Number of elements in feature array "
                         f"({n_elements}) is not the same as in "
                         f"the species list ({len(l_species)})!")

    # Build feature dictionary
    feat_vec_dict = {}
    for species_center in l_species:
        for species_neighbour in l_species:
            val = np.zeros(max_radial * max_angular)
            feat_vec_dict[species_center, species_neighbour] = val

    # Fill feature dictionary with each component pair
    for i_species_center in range(n_atoms):
        species_center = r_species[i_species_center]
        for i_species_neighbour in range(n_elements):
            species_neighbour = l_species[i_species_neighbour]
            val = frame_feature[i_species_center,
                                i_species_neighbour].flatten()
            feat_vec_dict[species_center, species_neighbour] += val

    return feat_vec_dict


def construct_feature_vector(X_raw, frames, l_species):
    """Construct feature vector for point chargees.

    Parameters
    ----------
    X_raw : np.array
        features of shape
        (n_frames, n_atoms, n_elements, max_radial, max_angular)
    frames : list
        ase trajectory
    l_species : list
        Element names corresponding to the elements in the third index of
        `X_raw`.

    Returns
    -------
    X : np.array
        features of shape (n_frames, n_elements**2 * max_radial * max_angular)
    """
    feature_shape = X_raw.shape
    n_frames = feature_shape[0]
    n_elements = feature_shape[2]
    max_radial = feature_shape[3]
    max_angular = feature_shape[4]

    if n_frames != len(frames):
        raise ValueError(f"Number of sets in feature array "
                         f"({n_frames}) is not the same as in "
                         f"the trajectory ({len(frames)})!")

    X = np.zeros([len(frames), n_elements**2 * max_radial * max_angular])

    for i in range(len(frames)):
        feat_vec_dict_pylode = _get_feat_vec_dict(l_species=l_species,
                                                  frame_feature=X_raw[i],
                                                  frame=frames[i])

        X[i] = np.hstack(feat_vec_dict_pylode.values())

    return X
