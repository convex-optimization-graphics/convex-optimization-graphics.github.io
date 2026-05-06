import numpy as np


def pmsdp_svd(P, Q):
    """
    Orthogonal Procrustes solve for corresponded point clouds. This is a Python
    implementation of procSvd.m from Maron et al. (2016).

    Given two (d, n) arrays P and Q with fixed correspondences, we solve
    min ||R*P - Q||_F over orthogonal R. Let Z = P*Q^T and write
    Z = U*Sigma*V^T from the SVD; the closed-form minimiser is R = V*U^T.

    Parameters
    ----------
    P, Q : (d, n) arrays
        Source and target points with known correspondences.

    Returns
    -------
    R : (d, d) array
        Orthogonal Procrustes solution.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    Z = P @ Q.T
    U, _, Vt = np.linalg.svd(Z)
    return Vt.T @ U.T
