import numpy as np
from scipy.optimize import linear_sum_assignment

from pmsdp_svd import proc_svd


def pmsdp_interleaving(X_init, R_init, P, Q, start_with="X", max_iter=200, tol=1e-6):
    """
    Refine a PM-SDP solution to a permutation and rotation by alternating updates.
    This is a Python implementation of interleaving.m from Maron et al. (2016),
    with the assignment step solved by scipy.optimize.linear_sum_assignment.

    At each iteration we alternate between two closed-form subproblems for
    min ||R*P - Q*X||_F: (i) orthogonal Procrustes for R with fixed X, and
    (ii) linear assignment for X with fixed R. With start_with='X', the update
    order is R then X; start_with='R' flips that order.

    Parameters
    ----------
    X_init : (n, n) array
        Initial correspondence matrix (need not be a valid permutation).
    R_init : (d, d) array
        Initial rotation (need not be exactly orthogonal).
    P, Q : (d, n) arrays
        Source and target point clouds.
    start_with : {'X', 'R'}
        Which variable to update first.
    max_iter : int, optional
        Maximum number of alternating iterations.
    tol : float, optional
        Stop when successive objective values differ by at most tol.

    Returns
    -------
    X : (n, n) array
        Recovered permutation matrix.
    R : (d, d) array
        Recovered rotation matrix.
    obj : float
        Final residual ||R*P - Q*X||_F.
    n_iter : int
        Number of iterations until convergence.
    """
    if start_with not in ("X", "R"):
        raise ValueError("start_with must be 'X' or 'R'.")
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    n = P.shape[1]

    X = X_init.copy()
    R = R_init.copy()
    prev_obj = np.inf
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        if start_with == "X":
            R = proc_svd(P, Q @ X)
            X = assign(P, Q, R, n)
        else:
            X = assign(P, Q, R, n)
            R = proc_svd(P, Q @ X)
        obj = float(np.linalg.norm(R @ P - Q @ X, "fro"))
        if abs(obj - prev_obj) <= tol:
            break
        prev_obj = obj
    return X, R, obj, n_iter


def assign(P, Q, R, n):
    """
    Solve the linear assignment step for fixed R in the interleaving loop.
    This maximises tr((P^T*R^T*Q)*X) over permutation matrices X.
    """
    cost = P.T @ R.T @ Q  # (n, n)
    row_ind, col_ind = linear_sum_assignment(-cost)
    X = np.zeros((n, n))
    X[col_ind, row_ind] = 1.0
    return X
