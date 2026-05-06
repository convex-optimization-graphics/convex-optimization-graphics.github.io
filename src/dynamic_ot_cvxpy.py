import numpy as np
import scipy.sparse as sp
import cvxpy as cp


def dynamic_ot_cvxpy(V, F, mu0, mu1, n_time, alpha=0.0, verbose=False):
    """
    Solve dynamical optimal transport on a triangle mesh via the dual SOCP. This is a Python 
    implementation of socpRun.m from Lavenant et al. (2018).

    FEM operators are assembled with unnormalised face areas to match the CVX and ADMM 
    conventions used in their MATLAB code. This is not the same convention as the EMD code...

    Parameters
    ----------
    V : (n_vertices, 3) array
        Vertex positions.
    F : (n_faces, 3) array
        Triangle indices.
    mu0, mu1 : (n_vertices,) arrays
        Source and target distributions in the ADMM convention
        (area-weighted mass, each summing to one).
    n_time : int
        Number of timesteps in the centred grid.
    alpha : float, optional
        Congestion regularization parameter. Use 0 for unregularized OT.
    verbose : bool, optional
        Pass through to the CVXPY solver.

    Returns
    -------
    mu_path : (n_time, n_vertices) array
        Interpolated density path on the centred time grid.
    objective : float
        Final dual objective value.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    nv, nf = V.shape[0], F.shape[0]

    mu0 = np.asarray(mu0, dtype=np.float64)
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu0 = mu0 / mu0.sum()
    mu1 = mu1 / mu1.sum()

    # Unnormalised face areas, per-component gradient and lumped vertex areas, matching firstOrderFEM.m
    face_vtx = [V[F[:, i]] for i in range(3)]
    edge_lengths = [np.linalg.norm(face_vtx[(i + 1) % 3] - face_vtx[i], axis=1) for i in range(3)]
    s = 0.5 * sum(edge_lengths)
    face_areas = np.sqrt(np.maximum(s * (s - edge_lengths[0]) * (s - edge_lengths[1]) * (s - edge_lengths[2]), 0.0))

    # grad[i] is the (n_f, n_v) sparse matrix that gives the i-th component
    # of grad(phi) on each face given vertex values phi
    N = np.cross(face_vtx[2] - face_vtx[0], face_vtx[1] - face_vtx[0])
    N = N / np.linalg.norm(N, axis=1, keepdims=True)
    grad_rows = [[], [], []]
    grad_cols = [[], [], []]
    grad_vals = [[], [], []]
    for i in range(3):
        j = (i + 1) % 3
        k = (j + 1) % 3
        d = face_vtx[k] - face_vtx[j]
        rot = np.cross(d, N)
        scaled = rot / (2.0 * face_areas[:, None])
        for comp in range(3):
            grad_rows[comp].append(np.arange(nf))
            grad_cols[comp].append(F[:, i])
            grad_vals[comp].append(scaled[:, comp])
    grad = [sp.csr_matrix((np.concatenate(grad_vals[c]), (np.concatenate(grad_rows[c]), np.concatenate(grad_cols[c]))), shape=(nf, nv)) for c in range(3)]

    # Lumped vertex areas mass per vertex: each face contributes area/3 to each of its three vertices
    area_weights = np.zeros(nv)
    for k in range(3):
        np.add.at(area_weights, F[:, k], face_areas / 3.0)

    # Spatial averaging matrix (n_v, n_f) row-normalised face-area weights
    rows = F.flatten()
    cols = np.repeat(np.arange(nf), 3)
    vals = np.repeat(face_areas, 3)
    spatial_avg = sp.csr_matrix((vals, (rows, cols)), shape=(nv, nf))
    row_sums = np.asarray(spatial_avg.sum(axis=1)).flatten()
    spatial_avg = sp.diags(1.0 / row_sums) @ spatial_avg

    # Time-derivative operator phi @ D.T gives (phi[:, t+1] - phi[:, t]) / tau
    tau = 1.0 / n_time
    D = sp.diags([-np.ones(n_time), np.ones(n_time)], offsets=[0, 1], shape=(n_time, n_time + 1)).tocsr() / tau

    alpha_cvx = 1.0 / alpha if alpha >= 1e-6 else 1e6

    phi = cp.Variable((nv, n_time + 1))
    norm_mtx = cp.Variable((nf, n_time + 1))
    lam = cp.Variable((nv, n_time))
    regterm = cp.Variable()

    time_deriv = phi @ D.T
    dx = [grad[i] @ phi for i in range(3)]
    grad_sq = cp.square(dx[0]) + cp.square(dx[1]) + cp.square(dx[2])

    time_avg = (norm_mtx[:, :n_time] + norm_mtx[:, 1:n_time + 1]) / 2
    spacetime_avg = spatial_avg @ time_avg

    hj_constraint = time_deriv + 0.5 * spacetime_avg <= lam
    constraints = [grad_sq <= norm_mtx, hj_constraint]

    if alpha >= 1e-6:
        weighted_lam_sq = cp.sum(cp.multiply(cp.square(lam), area_weights[:, None] / n_time))
        constraints.append(weighted_lam_sq <= regterm)
    else:
        constraints += [lam == 0, regterm == 0]

    obj = mu1 @ phi[:, n_time] - mu0 @ phi[:, 0] - 0.5 * alpha_cvx * regterm
    problem = cp.Problem(cp.Maximize(obj), constraints)
    last_err = None
    # this is all to avoid solver errors
    for solver in ("CLARABEL", "SCS"):
        if solver not in cp.installed_solvers():
            continue
        try:
            problem.solve(solver=solver, verbose=verbose)
            if problem.status in ("optimal", "optimal_inaccurate"):
                break
        except cp.error.SolverError as err:
            last_err = err
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"no solver returned an optimal status.")

    mu_dual = hj_constraint.dual_value
    mu_path = mu_dual * n_time
    return mu_path.T, float(problem.value)
