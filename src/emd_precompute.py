import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from src.first_order_fem import first_order_fem


def emd_precompute(V, F, curl_function_basis, harmonic_fields=None):
    """
    Precompute mesh structures used by the EMD ADMM solver. This is a Python implementation 
    of precomputeEarthMoversADMM.m. The expensive operators here depend only on the mesh 
    and chosen basis, so they can be reused across many solves.

    Parameters
    ----------
    V : (n_vertices, 3) array
        Vertex positions.
    F : (n_faces, 3) array
        Face indices.
    curl_function_basis : (n_vertices, n_curl) array
        Per-vertex scalar basis. Rotated gradients span the curl component of
        the flow.
    harmonic_fields : list of (n_faces, 3) arrays or None, optional
        Optional per-face harmonic vector fields for non-trivial cohomology.
        Use None for genus-0 surfaces.

    Returns
    -------
    precomp : dict
        FEM structure and matrices required by emd_admm.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    nv, nf = V.shape[0], F.shape[0]
    n_curl = curl_function_basis.shape[1]

    fem = first_order_fem(V, F)

    # B encodes the curl basis per face by stacking
    # area_f*grad(psi_k) into a (3*n_faces) vector for each k
    B = np.zeros((3 * nf, n_curl))
    for i in range(3):
        block = fem["face_inner_prods"] @ fem["grad"][i] @ curl_function_basis
        B[i::3, :] = np.asarray(block.todense() if sp.issparse(block) else block)

    n_harmonic = 0
    if harmonic_fields is not None:
        n_harmonic = len(harmonic_fields)
        face_normals_arr = face_normals(V, F)
        extra = np.zeros((3 * nf, n_harmonic))
        for k, hf in enumerate(harmonic_fields):
            rotated = np.cross(face_normals_arr, hf)
            for j in range(3):
                extra[j::3, k] = fem["face_areas"] * rotated[:, j]
        B = np.concatenate([B, extra], axis=1)

    least_squares_matrix = np.linalg.pinv(B.T @ B)

    # Sparse factorization of the negated Laplacian with first row and column pinned
    # to enforce the constant-mean gauge
    W = -fem["laplacian"].tolil()
    W[0, :] = 0.0
    W[:, 0] = 0.0
    W[0, 0] = 1.0
    W = W.tocsc()
    lu = spl.splu(W)

    face_normals_arr = face_normals(V, F)

    return {"fem": fem, "curl_function_basis": curl_function_basis, "n_harmonic": n_harmonic, "B": B,
            "least_squares_matrix": least_squares_matrix, "lu": lu, "face_normals": face_normals_arr}


def face_normals(V, F):
    """Compute unit face normals for a triangle mesh."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    norms[norms < np.finfo(float).eps] = 1.0
    return n / norms
