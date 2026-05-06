import numpy as np
import sympy as sp

def sos_symbolic_jacobian_det(V):
    """
    Python implementation of the symbolic_jacobian_det.m and symbolic_trilinear_map.m
    function from Marschner et al. (2020). The trilinear map sends (u, v, w) in [0, 1]^3
    to the interior of the hex with the eight given corner positions, in the
    standard hex ordering used by their function rand_hex.m:
        V[0] -> (0, 0, 0) V[4] -> (0, 0, 1)
        V[1] -> (1, 0, 0) V[5] -> (1, 0, 1)
        V[2] -> (1, 1, 0) V[6] -> (1, 1, 1)
        V[3] -> (0, 1, 0) V[7] -> (0, 1, 1)

    The Jacobian determinant is a polynomial in (u, v, w) with each variable
    appearing to degree at most two (total degree at most six).

    Parameters
    ----------
    V : (8, 3) array of vertex positions.

    Returns
    -------
    J : sympy.Poly
        The Jacobian determinant, expanded over (u, v, w). J(u, v, w) is
        returned as a sympy Poly for a single hex.
    """
    V = np.asarray(V, dtype=np.float64)
    if V.shape != (8, 3): raise ValueError(f"V must have shape (8, 3)")

    u, v, w = sp.symbols("u v w")
    shape_funcs = [(1 - u) * (1 - v) * (1 - w),
                    u * (1 - v) * (1 - w),
                    u * v * (1 - w),
                    (1 - u) * v * (1 - w),
                    (1 - u) * (1 - v) * w,
                    u * (1 - v) * w,
                    u * v * w,
                    (1 - u) * v * w]
    F = [sum(shape_funcs[i] * float(V[i, j]) for i in range(8)) for j in range(3)]
    jacobian = sp.Matrix([[sp.diff(F[i], var) for var in (u, v, w)] for i in range(3)])

    return sp.Poly(sp.expand(jacobian.det()), u, v, w)
