import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import igl


def rgd_admm(V, F, x0, reg='D', alpha_hat=0.1, beta_hat=0.0, vf=None,
            n_iter=10000, abs_tol=5e-6, rel_tol=1e-2, mu=10.0, 
            tau_inc=2.0, tau_dec=2.0, alpha_relax=1.7, verbose=False):
    """ADMM for regularized geodesic distances on a triangle mesh. This is a Python
    implementation of the function rgd_ADMM.m from "A Convex Optimization Framework 
    for Regularized Geodesic Distances" by Edelstein et al. (2023).

    Only support two of the regularizers from the paper, Dirichlet and vector-field 
    alignment.

    Parameters
    ----------
    V, F : array_like
        Mesh vertices and triangles.
    x0 : int or array of ints
        Source vertex index (or set of indices). The distance ``u`` is zero on
        the source and grows from there.
    reg : {'D', 'vfa'}
        Which regularizer to use.
    alpha_hat : float
        Scale-invariant regularisation weight. ``0`` is the unregularised Eikonal.
    beta_hat : float
        Vector-field alignment weight. Only used when reg='vfa'.
    vf : (n_faces, 3) array, optional
        Per-face line field to align isolines with. Only used when reg='vfa'.
    n_iter, abs_tol, rel_tol, mu, tau_inc, tau_dec, alpha_relax : int, float
        ADMM knobs.
    verbose : bool
        Print one line per iteration.

    Returns
    -------
    u : (n_vertices,) array_like
        Regularized geodesic distance from the source set.
    n_iter_used : int
        Number of ADMM iterations.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    nv, nf = V.shape[0], F.shape[0]
    x0 = np.atleast_1d(np.asarray(x0, dtype=np.int64))

    # Mesh operators
    Ww = -igl.cotmatrix(V, F).tocsr()  # positive-semidefinite cot Laplacian
    ta = 0.5 * igl.doublearea(V, F).astype(np.float64)
    f_to_v = sp.csr_matrix(
        (np.ones(3 * nf, dtype=np.float64),
         (np.repeat(np.arange(nf), 3), F.flatten())),
        shape=(nf, nv),
    )
    va = (f_to_v.T @ ta) / 3.0  # barycentric vertex areas
    G = igl.grad(V, F).tocsr()   # gradient (3 nf x nv)
    Ta3 = sp.diags(np.tile(ta, 3))

    # Build the stiffness matrix for each regularizer
    sum_va = float(va.sum())
    if reg == 'D':
        alpha = alpha_hat * np.sqrt(sum_va)
        Ww_s = None
        var_rho = True
    elif reg == 'vfa':
        if vf is None or np.asarray(vf).shape != (nf, 3):
            raise ValueError(f"reg='vfa' requires vf of shape ({nf}, 3); got {None if vf is None else np.asarray(vf).shape}")
        if np.linalg.norm(vf, axis=1).max() < 1e-10:
            raise ValueError("reg='vfa' got an all-zero vector field.")
        alpha = alpha_hat * np.sqrt(sum_va)
        beta = beta_hat * np.sqrt(sum_va)
        Vmat = vmat_outer_per_face(np.asarray(vf, dtype=np.float64), nf)
        Ww_s = G.T @ Ta3 @ (sp.identity(3 * nf) + beta * Vmat) @ G
        var_rho = False
    else:
        raise ValueError(f"reg must be 'D' or 'vfa'; got {reg!r}")

    # Eliminate the source rows and columns for Dirichlet bc u(x_0) = 0
    keep = np.setdiff1d(np.arange(nv), x0)
    Ww_p = Ww[keep, :][:, keep]
    G_p = G[:, keep]
    Gpt = G_p.T
    div_p = Gpt @ Ta3
    va_p = va[keep]
    n_kept = keep.shape[0]
    if Ww_s is not None:
        Ww_s_p = Ww_s[keep, :][:, keep]

    # ADMM state
    rho = 2.0 * np.sqrt(sum_va)
    u_p = np.zeros(n_kept)
    y = np.zeros(3 * nf)
    z = np.zeros(3 * nf)
    div_y = np.zeros(n_kept)
    div_z = np.zeros(n_kept)

    tasq = np.tile(np.sqrt(ta), 3)
    thresh1 = np.sqrt(3 * nf) * abs_tol * np.sqrt(sum_va)
    thresh2 = np.sqrt(nv) * abs_tol * sum_va

    # Pre-factorisation of the u-step system
    if reg == 'D':
        solve_u = splu_solver(Ww_p)
    elif not var_rho:
        solve_u = splu_solver(alpha * Ww_s_p + rho * Ww_p)

    n_iter_used = n_iter
    for it in range(1, n_iter + 1):
        b = va_p - div_y + rho * div_z
        if reg == 'D':
            u_p = solve_u(b) / (alpha + rho)
        else:
            u_p = solve_u(b)
        Gx = G_p @ u_p

        z_old = z
        div_z_old = div_z
        z = (1.0 / rho) * y + Gx
        z_face = z.reshape(3, nf).T
        norms = np.linalg.norm(z_face, axis=1)
        scale = np.maximum(norms, 1.0)
        z_face = z_face / scale[:, None]
        z = z_face.T.flatten()
        div_z = div_p @ z

        y = y + rho * (alpha_relax * Gx + (1.0 - alpha_relax) * z_old - z)
        div_y = div_p @ y

        tasqGx = tasq * Gx
        tasqZ = tasq * z
        r_norm = float(np.linalg.norm(tasqGx - tasqZ))
        s_norm = float(rho * np.linalg.norm(div_z - div_z_old))
        eps_pri = thresh1 + rel_tol * max(np.linalg.norm(tasqGx), np.linalg.norm(tasqZ))
        eps_dual = thresh2 + rel_tol * np.linalg.norm(div_y)

        if verbose:
            print(f'{it:4d}  r={r_norm:9.4f}  eps_pri={eps_pri:9.4f}  '
                  f's={s_norm:9.4f}  eps_dual={eps_dual:9.4f}')

        if it > 1 and r_norm < eps_pri and s_norm < eps_dual:
            n_iter_used = it
            break

        if var_rho:
            if r_norm > mu * s_norm:
                rho *= tau_inc
            elif s_norm > mu * r_norm:
                rho /= tau_dec

    u = np.zeros(nv)
    u[keep] = u_p
    return u, n_iter_used


def vmat_outer_per_face(vf, nf):
    """ Build the 3nf x 3nf block-diagonal of diagonals encoding outer products
    vf_face vf_face^T for every face. Matches the Vmat construction in their 
    MATLAB implementation"""
    rows = []
    for i in range(3):
        cols = [sp.diags(vf[:, i] * vf[:, j]) for j in range(3)]
        rows.append(sp.hstack(cols))
    return sp.vstack(rows).tocsr()


def splu_solver(A):
    lu = spl.splu(A.tocsc())
    return lu.solve
