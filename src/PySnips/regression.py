# -*- Mode: python3; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
"""Functions for linear and kernel regression."""

import numpy as np
import tqdm
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

from .statistics import get_rmse


def train_predict_lr(X,
                     Y,
                     i_train,
                     i_test,
                     r_train_structures,
                     regularisation=1e-10):
    """
    Train a linear model based on feature vector X.

    Parameters
    ----------
    X : ndarray of shape (n_sets_X, n_atoms, n_features)
        Feature array
    Y : ndarray of shape (n_sets_X)
        Observables for fitting
    i_train : ndarray
        indices of training sets
    i_test : ndarray
        indices of test sets
    r_train_structures : ndarray
        array containingt the number of training structures
    regularisation : float
        regularisation for regression

    Returns
    -------
    RMSE train : np.array
        root mean square deviation (%) of the training set
        with respect to the standard deviation of the test set.
    RMSE test : np.array
        root mean square deviation (%) of the test set
        with respect to the standard deviation of the test set.
    """
    X_train = X[i_train]
    X_test = X[i_test]

    Y_train = Y[i_train]
    Y_test = Y[i_test]

    rmse_train = np.zeros(len(r_train_structures))
    rmse_test = np.zeros(len(r_train_structures))

    for i, n_train_structure in enumerate(r_train_structures):
        X_train_cur = X_train[:n_train_structure]
        Y_train_cur = Y_train[:n_train_structure]

        clf = Ridge(alpha=regularisation)
        clf.fit(X_train_cur, Y_train_cur)

        Y_train_pred = clf.predict(X_train_cur)
        Y_test_pred = clf.predict(X_test)

        rmse_train[i] = get_rmse(Y_train_pred, Y_train_cur)
        rmse_test[i] = get_rmse(Y_test_pred, Y_test)

        # Calculate % RMSE
        rmse_train[i] *= 100 / Y_train_cur.var()
        rmse_test[i] *= 100 / Y_train_cur.var()

    return rmse_train, rmse_test


def build_kernel_matrix(kernel_func, X, Y, desc="Build kernel"):
    """Build kernel matrix for kernel ridge regression.

    Parameters
    ----------
    kernel_func : callable
        kernel function. i.e. `sklearn.metrics.pairwise.linear_kernel`
    X : ndarray of shape (n_sets_X, n_atoms, n_features)
        The first feature array.
    Y : ndarray of shape (n_sets_Y, n_atoms, n_features)
        The second feature array.
    desc : str
        Description for progress bar.

    Returns
    -------
    K : ndarray of shape (n_samples_X, n_samples_Y)

    Raises
    ------
    ValueError
        If predicted number of structures times `n_atoms` is not equal to
        the length of one of the feature arrays.
    """
    if X is Y:
        # For X==Y the kernel matrix is symmetric.
        # Only calculate upper trangle matrix and copy at the end.
        row, col = np.triu_indices(X.shape[0])
    else:
        row, col = np.indices((X.shape[0], Y.shape[0]))

    K = np.zeros([X.shape[0], Y.shape[0]])

    indices = [(r, c) for r, c in zip(row.flatten(), col.flatten())]
    for n, m in tqdm(indices, desc=desc):
        K[n, m] = np.sum(kernel_func(X[n], Y[m]))

    if X is Y:
        K += K.T
        K -= np.diag(np.diag(K)) / 2

    return K


def train_predict_krr(X,
                      Y,
                      i_train,
                      i_test,
                      r_train_structures,
                      kernel_func,
                      regularisation=1e-10):
    """
    Train a KRR model based on given feature vector X.

    Parameters
    ----------
    X : ndarray of shape (n_sets_X, n_atoms, n_features)
        Feature array
    Y : ndarray of shape (n_sets_X)
        Observables for fitting
    i_train : ndarray
        indices of training sets
    i_test : ndarray
        indices of test sets
    r_train_structures : ndarray
        array containingt the number of training structures
    kernel_func : callable
        kernel function. i.e. `sklearn.metrics.pairwise.linear_kernel`
    regularisation : float
        regularisation for regression

    Returns
    -------
    RMSE train : np.array
        root mean square deviation (%) of the training set
        with respect to the standard deviation of the test set.
    RMSE test : np.array
        root mean square deviation (%) of the test set
        with respect to the standard deviation of the test set.
    """
    X_train = X[i_train]
    X_test = X[i_test]

    Y_train = Y[i_train]
    Y_test = Y[i_test]

    rmse_train = np.zeros(len(r_train_structures))
    rmse_test = np.zeros(len(r_train_structures))

    for i, n_train_structure in enumerate(r_train_structures):
        X_train_cur = X_train[:n_train_structure]
        Y_train_cur = Y_train[:n_train_structure]

        K_train = build_kernel_matrix(kernel_func,
                                      X=X_train_cur,
                                      Y=X_train_cur,
                                      desc=f"Build train kernel for"
                                      f" {n_train_structure} sets")

        K_test = build_kernel_matrix(kernel_func,
                                     X=X_test,
                                     Y=X_train_cur,
                                     desc=f"Build test kernel for"
                                     f" {n_train_structure} sets")

        krr = KernelRidge(alpha=regularisation, kernel="precomputed", gamma=1)
        krr.fit(K_train, Y_train_cur)

        Y_train_pred = krr.predict(K_train)
        Y_test_pred = krr.predict(K_test)

        rmse_train[i] = get_rmse(Y_train_pred, Y_train_cur)
        rmse_test[i] = get_rmse(Y_test_pred, Y_test)

        # Calculate % RMSE
        rmse_train[i] *= 100 / Y_train_cur.var()
        rmse_test[i] *= 100 / Y_train_cur.var()

    return rmse_train, rmse_test
