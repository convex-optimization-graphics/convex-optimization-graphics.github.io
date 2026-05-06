import numpy as np
import sympy as sp
from SumOfSquares import SOSProblem, poly_variable

from src.sos_symbolic_jacobian_det import sos_symbolic_jacobian_det


def sos_jacobian_single_hex(V, k=4, verbose=False):
    """
    SOS relaxation for the minimum Jacobian determinant of a single hex. This is a 
    Python implementation of the SOS_jacobian_single_hex.m function from Marschner
    et al. (2020).
    
    The polynomial J(u, v, w) is the trilinear map's Jacobian determinant, and we want
    min_{(u, v, w) in [0, 1]^3} J(u, v, w). Putinar's positivstellensatz gives a non-
    negativity certificate J - bound = s_0 + s_1 u + s_2 (1 - u) + ... + s_6 (1 - w)
    with each ``s_i`` a sum of squares. Maximising bound over all such decompositions 
    is the SDP that the SumOfSquares call builds and Mosek solves.

    The dual of the slack SOS constraint is the moment matrix; its first column gives 
    the optimiser's coordinates (1, u^*, v^*, w^*, ...) when the relaxation is tight.

    Parameters
    ----------
    V : (8, 3) array
        Hex vertex positions.
    k : int, optional
        Degree of the SOS multipliers s_1,...,s_6. Must be even!
    verbose : bool, optional
        Pass through to the SDP solver.

    Returns
    -------
    bound : float
        SOS lower bound on min_{[0, 1]^3} J.
    argmin : (3,) numpy.ndarray
        Recovered argmin in parameter space, from the moment matrix.
    recovery_gap : float
        J(argmin) - bound. Near zero means the relaxation is tight and
        the recovered point is the true optimiser.
    """
    if k % 2 != 0:
        raise ValueError("k must be even (matches the YALMIP default 4).")

    u, v, w = sp.symbols("u v w")
    J = sos_symbolic_jacobian_det(V).as_expr()
    ineqs = [u, 1 - u, v, 1 - v, w, 1 - w]

    # SumOfSquares uses the convention deg = relaxation order. Each multiplier
    # s_i has degree (2*deg - deg(g_i))//2 * 2. With deg = (k + 2) // 2 we get
    # multiplier degree k, matching their MATLAB convention.
    deg = (k + 2) // 2

    prob = SOSProblem()
    gamma = sp.symbols("gamma")
    gamma_var = prob.sym_to_var(gamma)

    weighted_sum = 0
    for i, g in enumerate(ineqs):
        s = poly_variable(f"s{i}", [u, v, w], (2 * deg - 1) // 2 * 2)
        prob.add_sos_constraint(s, [u, v, w], name=f"s{i}")
        weighted_sum += s * g
    slack = prob.add_sos_constraint(sp.expand(J - gamma - weighted_sum), [u, v, w])
    prob.set_objective("max", gamma_var)
    prob.solve(solver="mosek", verbose=verbose)

    bound = float(prob.value)

    # Moment matrix is the dual of the slack SOS constraint, indexed by the
    # constraint's monomial basis. Normalise so the constant moment is one.
    n_basis = len(slack.basis.monoms)
    M = np.array(slack.pic_const.dual).reshape(n_basis, n_basis)
    M = M / M[0, 0]

    monoms = slack.basis.monoms
    argmin = np.array([float(M[monoms.index((1, 0, 0)), 0]),
                        float(M[monoms.index((0, 1, 0)), 0]),
                        float(M[monoms.index((0, 0, 1)), 0])])

    J_at_argmin = float(sp.lambdify((u, v, w), J, "numpy")(*argmin))
    return bound, argmin, J_at_argmin - bound
