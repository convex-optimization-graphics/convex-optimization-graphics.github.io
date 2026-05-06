import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import gpytoolbox as gpy
import igl


def bbw_demo(mesh_path, handle_positions):
    """
    Interactive bounded biharmonic weight demo.

    The QP solved at the start is the bounded biharmonic weights problem from
    Jacobson et al. (2011).

    Parameters
    ----------
    mesh_path : str
        Path to a triangle mesh.
    handle_positions : array-like, shape (n_handles, 3)
        Approximate target positions for the handles. Each is snapped to the
        nearest mesh vertex.
    """
    V_rest, F = gpy.read_mesh(str(mesh_path))
    V_rest = np.ascontiguousarray(V_rest[:, :3], dtype=np.float64)
    F = F.astype(np.int64)

    handle_positions = np.asarray(handle_positions, dtype=np.float64)
    handle_indices = np.array([int(np.argmin(np.linalg.norm(V_rest - h, axis=1))) for h in handle_positions], dtype=np.int64)
    n_handles = len(handle_indices)

    bc = np.eye(n_handles, dtype=np.float64)
    W = igl.bbw(V_rest, F, handle_indices, bc)
    handle_rest = V_rest[handle_indices].copy()
    translations = np.zeros((n_handles, 3), dtype=np.float64)

    ps.init()
    ps.remove_all_structures()
    ps.set_ground_plane_mode('none')
    ps_mesh = ps.register_surface_mesh('mesh', V_rest, F, smooth_shade=True)

    handle_clouds = []
    handle_names = []
    for j in range(n_handles):
        name = f'handle {j}'
        cloud = ps.register_point_cloud(name, handle_rest[j:j+1])
        cloud.set_radius(0.03)
        cloud.set_color((0.9, 0.2, 0.2))
        handle_clouds.append(cloud)
        handle_names.append(name)

    state = {'dragging': None, 'depth': None}

    def callback():
        io = psim.GetIO()
        psim.TextUnformatted('Click directly on a red handle and drag to translate it.')
        psim.TextUnformatted('Left-drag empty space rotates the camera as usual.')
        psim.Separator()

        # Mouse
        if io.WantCaptureMouse:
            if state['dragging'] is not None:
                state['dragging'] = None
                ps.set_do_default_mouse_interaction(True)
        else:
            mx, my = io.MousePos

            # Start drag if a handle was just clicked
            if io.MouseClicked[0] and state['dragging'] is None:
                result = ps.pick(screen_coords=(mx, my))
                if result.is_hit and result.structure_name in handle_names:
                    j = handle_names.index(result.structure_name)
                    cam = ps.get_view_camera_parameters()
                    look_dir = np.asarray(cam.get_look_dir(), dtype=np.float64)
                    cam_pos = np.asarray(cam.get_position(), dtype=np.float64)
                    handle_pos = handle_rest[j] + translations[j]
                    state['dragging'] = j
                    state['depth'] = float(np.dot(handle_pos - cam_pos, look_dir))
                    ps.set_do_default_mouse_interaction(False)

            # Continue drag
            if state['dragging'] is not None and io.MouseDown[0]:
                j = state['dragging']
                ray = np.asarray(ps.screen_coords_to_world_ray((mx, my)), dtype=np.float64)
                cam = ps.get_view_camera_parameters()
                look_dir = np.asarray(cam.get_look_dir(), dtype=np.float64)
                cam_pos = np.asarray(cam.get_position(), dtype=np.float64)
                denom = float(np.dot(ray, look_dir))
                if abs(denom) > 1e-8 and np.all(np.isfinite(ray)):
                    t = state['depth'] / denom
                    new_pos = cam_pos + t * ray
                    translations[j] = new_pos - handle_rest[j]
                    handle_clouds[j].update_point_positions(
                        (handle_rest[j] + translations[j]).reshape(1, 3)
                    )
                    ps_mesh.update_vertex_positions(V_rest + W @ translations)

            # End drag on release
            if io.MouseReleased[0] and state['dragging'] is not None:
                state['dragging'] = None
                ps.set_do_default_mouse_interaction(True)

        if psim.Button('reset handles'):
            translations[:] = 0.0
            for j, cloud in enumerate(handle_clouds):
                cloud.update_point_positions(handle_rest[j].reshape(1, 3))
            ps_mesh.update_vertex_positions(V_rest)

    ps.set_user_callback(callback)
    ps.show()
    ps.clear_user_callback()
    ps.set_do_default_mouse_interaction(True)
