import numpy as np
import scipy.sparse as sp


def first_order_fem(V, F):
    """
    Build first-order FEM operators on a triangle mesh. This is a Python implementation 
    of firstOrderFEM.m from Solomon et al. (2014). Face areas are normalized so total 
    surface area is one, matching the reference code.

    Parameters
    ----------
    V : (n_vertices, 3) array
        Vertex positions.
    F : (n_faces, 3) array
        Vertex indices per face.

    Returns
    -------
    fem : dict
        Dictionary with fields:
        face_areas : (n_faces,) array
            Face areas.
        vtx_inner_prods : (n_vertices, n_vertices) sparse matrix
            Mass matrix.
        face_inner_prods : (n_faces, n_faces) sparse matrix
            Face-area matrix.
        grad : list of 3 sparse matrices
            Each entry has shape (n_faces, n_vertices), one per component.
        laplacian : (n_vertices, n_vertices) sparse matrix
            Cotan Laplacian, negative semidefinite.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    nf = F.shape[0]
    nv = V.shape[0]

    face_vtx = [V[F[:, i]] for i in range(3)]
    edge_lengths = [np.linalg.norm(face_vtx[(i + 1) % 3] - face_vtx[i], axis=1) for i in range(3)]

    # Heron's formula, then normalize so the surface integrates to 1
    s = 0.5 * (edge_lengths[0] + edge_lengths[1] + edge_lengths[2])
    face_areas = np.sqrt(np.maximum(s * (s - edge_lengths[0]) * (s - edge_lengths[1]) * (s - edge_lengths[2]), 0.0))
    total_area = face_areas.sum()
    face_areas = face_areas / total_area

    # Vertex mass matrix M:
    # M_ii = sum_f area_f / 6 over incident faces,
    # M_ij = sum_f area_f / 12 over faces containing edge (i, j)
    rows, cols, vals = [], [], []
    for i in range(3):
        j = (i + 1) % 3
        t1, t2 = F[:, i], F[:, j]
        rows.extend([t1, t2, t1])
        cols.extend([t2, t1, t1])
        vals.extend([face_areas / 12.0, face_areas / 12.0, face_areas / 6.0])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    vtx_inner_prods = sp.csr_matrix((vals, (rows, cols)), shape=(nv, nv))

    face_inner_prods = sp.diags(face_areas)

    # Per-face gradient: grad(phi) = sum_i phi_i*grad(hat_i)
    # For each face, grad(hat_i) = rot90(edge_jk)/(2*area), edge_jk = V_k - V_j
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
        scaled = rot / np.sqrt(total_area) / (2.0 * face_areas[:, None])
        for comp in range(3):
            grad_rows[comp].append(np.arange(nf))
            grad_cols[comp].append(F[:, i])
            grad_vals[comp].append(scaled[:, comp])

    grad = []
    for comp in range(3):
        rr = np.concatenate(grad_rows[comp])
        cc = np.concatenate(grad_cols[comp])
        vv = np.concatenate(grad_vals[comp])
        grad.append(sp.csr_matrix((vv, (rr, cc)), shape=(nf, nv)))

    # Cotan Laplacian, negative semidefinite
    laplacian = -(grad[0].T @ face_inner_prods @ grad[0]
                  + grad[1].T @ face_inner_prods @ grad[1]
                  + grad[2].T @ face_inner_prods @ grad[2])

    return {"face_areas": face_areas, "vtx_inner_prods": vtx_inner_prods,
        "face_inner_prods": face_inner_prods, "grad": grad, "laplacian": laplacian}
