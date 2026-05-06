import os
import sys
import pathlib
import numpy as np
from numpy import linalg

def dynamic_ot_run(mesh_path, boundary_path, n_time=31, congestion=0.0, n_iter=300, tol=1e-4, verbose=False):
    """
    Run the dynamical OT ADMM solver from external/DynamicalOTSurfaces. This is a just a 
    wrapper around the geodesic solver of Lavenant et al. (2018).

    Parameters
    ----------
    mesh_path : str or pathlib.Path
        Path to a .off mesh file.
    boundary_path : str or pathlib.Path
        Path to a .bdy file (valid Python) defining mub0 and mub1.
    n_time : int
        Number of time discretization points.
    congestion : float
        Congestion parameter. Use 0 for plain OT.
    n_iter : int
        Maximum number of ADMM iterations.
    tol : float
        Primal/dual feasibility tolerance.
    verbose : bool, optional
        Pass through to the solver.

    Returns
    -------
    result : dict
        V : (n_vertices, 3) 
            Vertex positions.
        F : (n_faces, 3) 
            Face indices.
        mu : (n_time, n_vertices) 
            Time-varying density.
        objective : 
            Objective values per iteration.
        primal_residual, dual_residual : 
            ADMM diagnostics.
    """
    repo = pathlib.Path(__file__).resolve().parent.parent
    admm_dir = repo / 'external' / 'DynamicalOTSurfaces' / 'ADMM code'
    if not admm_dir.is_dir():
        raise FileNotFoundError(f"DynOT submodule not found at {admm_dir}. Did you run `git submodule update --init --recursive`?")

    sys_path_added = False
    if str(admm_dir) not in sys.path:
        sys.path.insert(0, str(admm_dir))
        sys_path_added = True

    try:
        import read_off
        import surface_pre_computations
        from geodesic_surface_congested import geodesic
        import cut_off
    finally:
        # clean up sys.path to avoid issues
        if sys_path_added:
            sys.path.remove(str(admm_dir))

    mesh_path = str(mesh_path)
    boundary_path = str(boundary_path)

    Vertices, Triangles, Edges = read_off.readOff(mesh_path)
    areaTriangles, _, _ = surface_pre_computations.geometricQuantities(Vertices, Triangles, Edges)
    _, areaVertices, _ = surface_pre_computations.trianglesToVertices(Vertices, Triangles, areaTriangles)

    # The .bdy script expects Vertices, areaVertices, np, lin, cut_off in scope
    with open(boundary_path) as f:
        boundary_code = f.read()
    local_ns = {'Vertices': Vertices, 'areaVertices': areaVertices, 'np': np, 'lin': linalg, 'cut_off': sys.modules.get('cut_off')}
    # Change into the .bdy file's parent so relative paths resolve correctly
    bdy_parent_parent = pathlib.Path(boundary_path).resolve().parent.parent
    cwd_save = os.getcwd()
    try:
        os.chdir(bdy_parent_parent)
        exec(boundary_code, local_ns)
    finally:
        os.chdir(cwd_save)
    mub0 = local_ns['mub0']
    mub1 = local_ns['mub1']

    eps = 0.0
    detail_study = False
    phi, mu, A, E, B, objective, primal, dual = geodesic(n_time, mesh_path, mub0, mub1, congestion, \
                                                         eps, n_iter, detail_study, verbose, tol)

    return {'V': np.asarray(Vertices, dtype=np.float64), 'F': np.asarray(Triangles, dtype=np.int64), 
            'mu': np.asarray(mu, dtype=np.float64), 'objective': np.asarray(objective, dtype=np.float64), 
            'primal_residual': np.asarray(primal, dtype=np.float64), 'dual_residual': np.asarray(dual, dtype=np.float64)}
