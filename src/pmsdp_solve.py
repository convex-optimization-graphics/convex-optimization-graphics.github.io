import numpy as np
import cvxpy as cp

from src.pmsdp_interleaving import pmsdp_interleaving


def pmsdp_solve(P, Q, R_band=None, verbose=False):
    """
    SDP relaxation of the Procrustes Matching problem (PM-SDP). This is a Python
    implementation of solvePMSDP.m and generateConstraints.m from Maron et al.
    (2016).

    Given source and target (d, n) point clouds P and Q of the same shape, we seek a
    rotation R and permutation X minimising ||R*P - Q*X||_F^2. The non-convex
    formulation is lifted to an SDP with per-column PSD blocks:

    M_i = [[1, X[:, i]^T, vec(R)^T], [X[:, i], diag(X[:, i]), Y_i], [vec(R), Y_i^T, B]] >> 0,

    plus doubly-stochastic constraints on X, consistency sum(Y_i, axis=0) = vec(R),
    and lifted orthogonality constraints on B encoding R^T*R = R*R^T = I in vectorised
    form. The SDP solution is generally not exactly permutation/orthogonal, so they
    post-process it with pmsdp_interleaving.

    Parameters
    ----------
    P, Q : (d, n) arrays
        Source / target points.
    R_band : int or None, optional
        If set, restrict R to a (2 R_band + 1)-band around the diagonal, i.e.
        R[i, j] = 0 whenever |i - j| > R_band. This mirrors the utilizeRFlag / Rtol
        setting in the original code for low-frequency functional-map priors.
    verbose : bool, optional
        Pass through to the SDP solver.

    Returns
    -------
    X_proj : (n, n) array
        Recovered permutation matrix.
    R_proj : (d, d) array
        Recovered rotation matrix.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    d, n = P.shape
    if Q.shape != (d, n):
        raise ValueError("P and Q must have the same shape.")

    norm_P_sq = float(np.linalg.norm(P, "fro") ** 2)
    norm_Q_per_col_sq = np.sum(Q ** 2, axis=0)
    W = np.kron(P, Q)  # (d^2, n^2)

    X = cp.Variable((n, n), nonneg=True)
    R = cp.Variable((d, d))
    B = cp.Variable((d ** 2, d ** 2), symmetric=True)
    Y = [cp.Variable((n, d ** 2)) for _ in range(n)]

    constraints = []

    # Per-column PSD block of size (1 + n + d^2)
    for i in range(n):
        Xi = cp.reshape(X[:, i], (n, 1), order="F")
        rvec = cp.reshape(R, (d ** 2, 1), order="F")
        block = cp.bmat([
            [np.ones((1, 1)), Xi.T, rvec.T],
            [Xi, cp.diag(X[:, i]), Y[i]],
            [rvec, Y[i].T, B],
        ])
        constraints.append(block >> 0)
        constraints.append(cp.sum(Y[i], axis=0) == cp.reshape(R, (d ** 2,), order="F"))

    # Doubly-stochastic constraints on X
    constraints += [cp.sum(X, axis=0) == 1, cp.sum(X, axis=1) == 1]

    # Optional R-banding: zero entries outside the diagonal band in R, and the
    # corresponding rows/columns in lifted variables indexed by vec(R)
    if R_band is not None:
        for j in range(d):
            for i in range(d):
                if abs(i - j) > R_band:
                    constraints.append(R[i, j] == 0)
                    cm = i + j * d
                    constraints.append(B[cm, :] == 0)
                    constraints.append(B[:, cm] == 0)
                    for col in range(n):
                        constraints.append(Y[col][:, cm] == 0)

    # Lifted rotation constraints for R^T*R = I and R*R^T = I in vec(R) form
    for j in range(d):
        for k in range(d):
            entries = [B[i + j * d, i + k * d] for i in range(d)]
            constraints.append(sum(entries) == (1.0 if j == k else 0.0))

    for i in range(d):
        for k in range(d):
            entries = [B[i + j * d, k + j * d] for j in range(d)]
            constraints.append(sum(entries) == (1.0 if i == k else 0.0))

    # Objective corresponding to the PM-SDP lift
    obj_expr = 0
    for i in range(n):
        W_i = W[:, i * n : (i + 1) * n]  # (d^2, n)
        obj_expr = obj_expr + 2 * cp.trace(W_i @ Y[i])
    obj_expr = obj_expr - norm_P_sq - norm_Q_per_col_sq @ cp.sum(X, axis=1)

    problem = cp.Problem(cp.Maximize(obj_expr), constraints)
    problem.solve(solver=cp.MOSEK, verbose=verbose)

    X_val = np.asarray(X.value)
    R_val = np.asarray(R.value)

    # Rank-one factor of stacked Y for the upper-bound initialization
    Y_stack = np.vstack([np.asarray(Yi.value) for Yi in Y])  # (n^2, d^2)
    U_y, _, Vt_y = np.linalg.svd(Y_stack, full_matrices=False)
    XY = U_y[:, 0].reshape(n, n)
    RY = Vt_y[0, :].reshape(d, d, order="F")

    # Try lower-/upper-bound seeds and keep the best interleaving result
    best_X, best_R, best_obj = None, None, np.inf
    for X_start, R_start in ((X_val, R_val), (XY, RY)):
        for which in ("X", "R"):
            Xp, Rp, obj_val, _ = pmsdp_interleaving(X_start, R_start, P, Q, start_with=which)
            if obj_val < best_obj:
                best_obj = obj_val
                best_X, best_R = Xp, Rp

    return best_X, best_R
