import numpy as np


def emd_admm(rho0, rho1, structure, rho=10000.0, n_iter=100, tol=1e-5, verbose=False):
    """
    Solve EMD between two distributions on a triangle mesh. This is a Python implementation 
    of the ADMM solver from Solomon et al. (2014).

    The flow field J is decomposed as J = grad(a) + R grad(b), where R rotates 90 degrees 
    in each tangent plane. The grad part a is fixed by Laplacian(a) = rho1 - rho0, a closed
    form, and ADMM is run on b to minimise sum_f area_f * ||J_f||_2.

    Parameters
    ----------
    rho0, rho1 : (n_vertices,) array_like
        Per-vertex distributions. Both should sum to 1. Assumed already weighted by vertex areas.
    structure : dict
        Dictionary from emd_precompute.
    rho : float
        ADMM penalty parameter. The default 10000 matches the MATLAB code.
    n_iter : int
        Maximum number of ADMM iterations.
    tol : float
        Stop early when the change in the dual variable c falls below this.
    verbose : bool
        If True, print one line per iteration.

    Returns
    -------
    distance : float
        EMD value
    J : (n_faces, 3) 
        Optimal momentum vector field.
    """
    rho0 = np.asarray(rho0, dtype=np.float64).reshape(-1)
    rho1 = np.asarray(rho1, dtype=np.float64).reshape(-1)
    fem = structure["fem"]
    B = structure["B"]
    nf = fem["face_areas"].shape[0]
    n_curl = structure["curl_function_basis"].shape[1] + structure["n_harmonic"]

    # Solve Laplacian(a) = rho1 - rho0 via the prefactored Cholesky
    rhs = (rho1 - rho0).copy()
    rhs[0] = 0.0
    a = -structure["lu"].solve(rhs)

    # Centre a so that its weighted mean is zero
    integral = float((fem["vtx_inner_prods"] @ a).sum())
    total_area = float(fem["vtx_inner_prods"].sum())
    a = a - integral / total_area

    # Per-face gradient of a. The divergence-free residual will be the curl part.
    grad_part = np.column_stack([fem["grad"][i] @ a for i in range(3)])

    # The ADMM works on the rotated gradient field w
    rot_grad = np.cross(structure["face_normals"], grad_part)
    w = (fem["face_areas"][:, None] * rot_grad).reshape(-1)

    c = np.zeros(n_curl)
    y = np.zeros(3 * nf)
    Bc = np.zeros(3 * nf)
    pp = structure["least_squares_matrix"] @ B.T

    for it in range(n_iter):
        z = Bc + w - y / rho

        # J step
        z_face = z.reshape(nf, 3)
        norms = rho * np.linalg.norm(z_face, axis=1)
        scale = np.where(norms > 4.0, 1.0 - 4.0 / norms, 0.0)
        J_step = (scale[:, None] * z_face).reshape(-1)

        # c step
        old_c = c
        c = pp @ (y / rho + J_step - w)

        if it > 3 and np.sum((c - old_c) ** 2) < tol:
            if verbose:
                print(f"converged at iteration {it + 1}")
            break

        Bc = B @ c
        y = y + rho * (J_step - Bc - w)
        if verbose:
            print(f"ADMM iter {it + 1}")

    # Recover J = grad(a) + R grad(b) and the EMD value
    n_curl_only = structure["curl_function_basis"].shape[1]
    curl_part = np.column_stack([fem["grad"][i] @ (structure["curl_function_basis"] @ c[:n_curl_only]) for i in range(3)])
    curl_part = np.cross(curl_part, structure["face_normals"])
    J = curl_part + grad_part

    resid = (B @ c + w).reshape(nf, 3)
    distance = float(np.linalg.norm(resid, axis=1).sum())
    return distance, J
