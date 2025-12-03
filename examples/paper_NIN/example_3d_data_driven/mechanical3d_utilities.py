import jax.numpy as jnp
import numpy as np
from fol.mesh_input_output.mesh import Mesh
import gmsh, os, math, random, meshio
from fol.tools.usefull_functions import *

def create_gyroid(fe_mesh: Mesh, tpms_settings: dict):
    """
    section: dict with keys:
        - "axis of section": "z"
        - "value": float or int, the coordinate at which to take the slice
    """

    phi_x = tpms_settings["phi_x"]
    phi_y = tpms_settings["phi_y"]
    phi_z = tpms_settings["phi_z"]
    max = tpms_settings["max"]
    min = tpms_settings["min"]
    threshold = tpms_settings["threshold"]
    fx, fy, fz = tpms_settings["coefficients"]
    if tpms_settings.get("constant") is not None:
        const = tpms_settings["constant"]
    else:
        const = 0.

    x = fe_mesh.GetNodesX()
    y = fe_mesh.GetNodesY()
    z = fe_mesh.GetNodesZ()

    if tpms_settings.get("section_axis_value") is not None:
        value = tpms_settings["section_axis_value"]
        z = value * jnp.ones_like(x)

    cos_pix = jnp.cos(fx * jnp.pi * x + phi_x)
    sin_pix = jnp.sin(fx * jnp.pi * x + phi_x)
    cos_piy = jnp.cos(fy * jnp.pi * y + phi_y)
    sin_piy = jnp.sin(fy * jnp.pi * y + phi_y)
    cos_piz = jnp.cos(fz * jnp.pi * z + phi_z)
    sin_piz = jnp.sin(fz * jnp.pi * z + phi_z)

    K = cos_piy * sin_pix + cos_piz * sin_piy + cos_pix * sin_piz - const
    binary_K = jnp.where((K < threshold) & (K > -threshold), max, min)

    return binary_K

def create_schwarz_P(fe_mesh: Mesh, tpms_settings: dict):
    """
    section: dict with keys:
        - "axis of section": "z"
        - "value": float or int, the coordinate at which to take the slice
    """

    phi_x = tpms_settings["phi_x"]
    phi_y = tpms_settings["phi_y"]
    phi_z = tpms_settings["phi_z"]
    max = tpms_settings["max"]
    min = tpms_settings["min"]
    threshold = tpms_settings["threshold"]
    fx, fy, fz = tpms_settings["coefficients"]
    if tpms_settings.get("constant") is not None:
        const = tpms_settings["constant"]
    else:
        const = 0.

    x = fe_mesh.GetNodesX()
    y = fe_mesh.GetNodesY()
    z = fe_mesh.GetNodesZ()

    if tpms_settings.get("section_axis_value") is not None:
        value = tpms_settings["section_axis_value"]
        z = value * jnp.ones_like(x)

    cos_pix = jnp.cos(fx * jnp.pi * x + phi_x)
    cos_piy = jnp.cos(fy * jnp.pi * y + phi_y)
    cos_piz = jnp.cos(fz * jnp.pi * z + phi_z)


    K = cos_piy + cos_piz + cos_pix - const
    binary_K = jnp.where((K < threshold) & (K > -threshold), max, min)

    return binary_K

def create_diamond(fe_mesh: Mesh, tpms_settings: dict):
    """
    section: dict with keys:
        - "axis of section": "z"
        - "value": float or int, the coordinate at which to take the slice
    """

    phi_x = tpms_settings["phi_x"]
    phi_y = tpms_settings["phi_y"]
    phi_z = tpms_settings["phi_z"]
    max = tpms_settings["max"]
    min = tpms_settings["min"]
    threshold = tpms_settings["threshold"]
    fx, fy, fz = tpms_settings["coefficients"]
    if tpms_settings.get("constant") is not None:
        const = tpms_settings["constant"]
    else:
        const = 0.

    x = fe_mesh.GetNodesX()
    y = fe_mesh.GetNodesY()
    z = fe_mesh.GetNodesZ()

    if tpms_settings.get("section_axis_value") is not None:
        value = tpms_settings["section_axis_value"]
        z = value * jnp.ones_like(x)

    cos_pix = jnp.cos(fx * jnp.pi * x + phi_x)
    cos_piy = jnp.cos(fy * jnp.pi * y + phi_y)
    cos_piz = jnp.cos(fz * jnp.pi * z + phi_z)
    sin_pix = jnp.sin(fx * jnp.pi * x + phi_x)
    sin_piy = jnp.sin(fy * jnp.pi * y + phi_y)
    sin_piz = jnp.sin(fz * jnp.pi * z + phi_z)



    K = sin_pix * sin_piy * sin_piz +\
          sin_pix * cos_piy * cos_piz +\
              cos_pix * sin_piy * cos_piz +\
                  cos_pix * cos_piy * sin_piz - const
    binary_K = jnp.where((K < threshold) & (K > -threshold), max, min)

    return binary_K

def create_lidinoid(fe_mesh: Mesh, tpms_settings: dict):
    """
    section: dict with keys:
        - "axis of section": "z"
        - "value": float or int, the coordinate at which to take the slice
    """

    phi_x = tpms_settings["phi_x"]
    phi_y = tpms_settings["phi_y"]
    phi_z = tpms_settings["phi_z"]
    max = tpms_settings["max"]
    min = tpms_settings["min"]
    threshold = tpms_settings["threshold"]
    fx, fy, fz = tpms_settings["coefficients"]
    if tpms_settings.get("constant") is not None:
        const = tpms_settings["constant"]
    else:
        const = -0.3

    x = fe_mesh.GetNodesX()
    y = fe_mesh.GetNodesY()
    z = fe_mesh.GetNodesZ()

    if tpms_settings.get("section_axis_value") is not None:
        value = tpms_settings["section_axis_value"]
        z = value * jnp.ones_like(x)

    cos_pix = jnp.cos(fx * jnp.pi * x + phi_x)
    cos_piy = jnp.cos(fy * jnp.pi * y + phi_y)
    cos_piz = jnp.cos(fz * jnp.pi * z + phi_z)
    sin_pix = jnp.sin(fx * jnp.pi * x + phi_x)
    sin_piy = jnp.sin(fy * jnp.pi * y + phi_y)
    sin_piz = jnp.sin(fz * jnp.pi * z + phi_z)
    cos_2pix = jnp.cos(2*(fx * jnp.pi * x + phi_x))
    cos_2piy = jnp.cos(2*(fy * jnp.pi * y + phi_y))
    cos_2piz = jnp.cos(2*(fz * jnp.pi * z + phi_z))
    sin_2pix = jnp.sin(2*(fx * jnp.pi * x + phi_x))
    sin_2piy = jnp.sin(2*(fy * jnp.pi * y + phi_y))
    sin_2piz = jnp.sin(2*(fz * jnp.pi * z + phi_z))



    K = sin_2pix * cos_piy * sin_piz +\
          sin_pix * sin_2piy * cos_piz +\
              cos_pix * sin_piy * sin_2piz -\
                  cos_2pix * cos_2piy - cos_2piy * cos_2piz - cos_2piz * cos_2pix - const
    binary_K = jnp.where((K < threshold) & (K > -threshold), max, min)

    return binary_K

def create_split_p(fe_mesh: Mesh, tpms_settings: dict):
    """
    section: dict with keys:
        - "axis of section": "z"
        - "value": float or int, the coordinate at which to take the slice
    """

    phi_x = tpms_settings["phi_x"]
    phi_y = tpms_settings["phi_y"]
    phi_z = tpms_settings["phi_z"]
    max = tpms_settings["max"]
    min = tpms_settings["min"]
    threshold = tpms_settings["threshold"]
    fx, fy, fz = tpms_settings["coefficients"]

    x = fe_mesh.GetNodesX()
    y = fe_mesh.GetNodesY()
    z = fe_mesh.GetNodesZ()

    if tpms_settings.get("section_axis_value") is not None:
        value = tpms_settings["section_axis_value"]
        z = value * jnp.ones_like(x)

    cos_pix = jnp.cos(fx * jnp.pi * x + phi_x)
    cos_piy = jnp.cos(fy * jnp.pi * y + phi_y)
    cos_piz = jnp.cos(fz * jnp.pi * z + phi_z)
    sin_pix = jnp.sin(fx * jnp.pi * x + phi_x)
    sin_piy = jnp.sin(fy * jnp.pi * y + phi_y)
    sin_piz = jnp.sin(fz * jnp.pi * z + phi_z)
    cos_2pix = jnp.cos(2*(fx * jnp.pi * x + phi_x))
    cos_2piy = jnp.cos(2*(fy * jnp.pi * y + phi_y))
    cos_2piz = jnp.cos(2*(fz * jnp.pi * z + phi_z))
    sin_2pix = jnp.sin(2*(fx * jnp.pi * x + phi_x))
    sin_2piy = jnp.sin(2*(fy * jnp.pi * y + phi_y))
    sin_2piz = jnp.sin(2*(fz * jnp.pi * z + phi_z))



    K = 1.1*(sin_2pix * cos_piy * sin_piz +
            sin_pix * sin_2piy * cos_piz +
            cos_pix * sin_piy * sin_2piz) - 0.2*(
        cos_2pix * cos_2piy + 
            cos_2piy * cos_2piz + 
            cos_2piz * cos_2pix) - 0.4*(
        cos_2pix + cos_2piy + cos_2piz)
    binary_K = jnp.where((K < threshold) & (K > -threshold), max, min)

    return binary_K


def create_random_periodic_sphere_field(fe_mesh, tpms_settings):
    """
    Generate a periodic field with randomly distributed non-overlapping spheres.

    Parameters:
    - fe_mesh: finite element mesh with node positions.
    - tpms_settings: dict with:
        - "sphere_diameter": float
        - "min": value inside spheres
        - "max": value outside spheres
        - "tolerance": optional fuzzy boundary width
        - "num_spheres": optional int, default=30
    """

    D = tpms_settings["sphere_diameter"]
    r = D / 2.0
    tol = tpms_settings.get("tolerance", 0.02 * D)
    min_val = tpms_settings["min"]
    max_val = tpms_settings["max"]
    num_spheres = tpms_settings.get("num_spheres", 30)

    # Mesh node coordinates
    if tpms_settings.get("section_axis_value") is not None:
        # FE init starts here
        N = int(fe_mesh.GetNumberOfNodes()**0.5)
        # Generate mesh coordinates
        x = jnp.linspace(0, 1, N)
        y = jnp.linspace(0, 1, N)
        z = jnp.linspace(0, 1, N)
        X, Y, Z = jnp.meshgrid(x, y, z)
        x = X.flatten()
        y = Y.flatten()
        z = Z.flatten()
        coords = jnp.stack([x, y, z], axis=1)
    else:
        x = fe_mesh.GetNodesX()
        y = fe_mesh.GetNodesY()
        z = fe_mesh.GetNodesZ()
        coords = jnp.stack([x, y, z], axis=1)

    # Generate non-overlapping centers
    centers = []
    max_attempts = 1000
    attempts = 0
    while len(centers) < num_spheres and attempts < max_attempts:
        candidate = jnp.array(np.random.rand(3))  # random in [0, 1]^3
        is_far_enough = True
        for c in centers:
            delta = jnp.abs(candidate - c)
            delta = jnp.minimum(delta, 1.0 - delta)  # periodic
            dist = jnp.linalg.norm(delta)
            if dist < D:  # overlapping
                is_far_enough = False
                break
        if is_far_enough:
            centers.append(candidate)
        attempts += 1

    centers = jnp.array(centers)

    # Compute minimum distance from each node to any sphere center (periodic)
    dist_min = jnp.full((coords.shape[0],), jnp.inf)
    for center in centers:
        delta = jnp.abs(coords - center)
        delta = jnp.minimum(delta, 1.0 - delta)  # periodic wrapping
        dist = jnp.linalg.norm(delta, axis=1)
        dist_min = jnp.minimum(dist_min, dist)

    # Thresholding
    inside = dist_min <= (r + tol)
    binary_K = jnp.where(inside, min_val, max_val)

    if tpms_settings.get("section_axis_value") is not None:
        z_val = tpms_settings.get("section_axis_value")
        Nz = len(jnp.unique(z))
        mask = jnp.isclose(z, z_val, atol=1.0 / Nz / 2)
        K_slice = binary_K[mask]
        return K_slice

    return binary_K

def create_random_fiber_field(fe_mesh, tpms_settings):
    """
    Create a field with elongated super-ellipsoids (fibers) inside a periodic box.
    
    Parameters:
    - fe_mesh: provides node positions.
    - tpms_settings: dict with:
        - "fiber_length": float (along main axis)
        - "fiber_radius": float (minor radius)
        - "min", "max": float
        - "num_fibers": int
        - "tolerance": optional, fuzzy boundary width
    """
    L = tpms_settings["fiber_length"]
    r = tpms_settings["fiber_radius"]
    min_val = tpms_settings["min"]
    max_val = tpms_settings["max"]
    tol = tpms_settings.get("tolerance", 0.02 * r)
    num_fibers = tpms_settings.get("num_fibers", 30)


    # Mesh node coordinates
    if tpms_settings.get("section_axis_value") is not None:
        # FE init starts here
        N = int(fe_mesh.GetNumberOfNodes()**0.5)
        # Generate mesh coordinates
        x = jnp.linspace(0, 1, N)
        y = jnp.linspace(0, 1, N)
        z = jnp.linspace(0, 1, N)
        X, Y, Z = jnp.meshgrid(x, y, z)
        x = X.flatten()
        y = Y.flatten()
        z = Z.flatten()
        coords = jnp.stack([x, y, z], axis=1)
    else:
        x = fe_mesh.GetNodesX()
        y = fe_mesh.GetNodesY()
        z = fe_mesh.GetNodesZ()
        coords = jnp.stack([x, y, z], axis=1)

    # Each fiber has center + random unit direction
    centers = []
    directions = []
    max_attempts = 1000
    attempts = 0

    while len(centers) < num_fibers and attempts < max_attempts:
        center = np.random.rand(3)
        theta = np.random.uniform(0, jnp.pi)
        phi = np.random.uniform(0, 2 * jnp.pi)
        direction = jnp.array([
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta)
        ])
        # Skip overlap checking for simplicity (can be added)
        centers.append(jnp.array(center))
        directions.append(direction)
        attempts += 1

    centers = jnp.stack(centers)
    directions = jnp.stack(directions)

    # Distance function for super-ellipsoid (fiber)
    dist_min = jnp.full((coords.shape[0],), jnp.inf)
    for i in range(num_fibers):
        c = centers[i]
        d = directions[i]
        vec = coords - c
        vec = jnp.where(vec > 0.5, vec - 1.0, vec)
        vec = jnp.where(vec < -0.5, vec + 1.0, vec)

        # Project onto fiber axis
        t = jnp.dot(vec, d)
        t = jnp.clip(t, -L/2, L/2)
        proj = jnp.outer(t, d)
        radial = vec - proj
        radial_dist = jnp.linalg.norm(radial, axis=1)

        # inside if radial <= r and |t| <= L/2
        inside = (radial_dist <= (r + tol)) & (jnp.abs(t) <= (L/2 + tol))
        dist_min = jnp.where(inside, 0.0, dist_min)

    binary_K = jnp.where(dist_min == 0.0, min_val, max_val)

    if tpms_settings.get("section_axis_value") is not None:
        z_val = tpms_settings.get("section_axis_value")
        Nz = len(jnp.unique(z))
        mask = jnp.isclose(z, z_val, atol=1.0 / Nz / 2)
        K_slice = binary_K[mask]
        return K_slice
    
    return binary_K




def create_cube_with_spheres_mesh(num_spheres, Lx, Ly, Lz, case_dir,
                                  min_radius=0.05, max_radius=0.15,
                                  mesh_size_min=None, mesh_size_max=None,
                                  seed=42):
    """
    Create a tetrahedral mesh of a cube with spherical voids.

    Special placements:
        num_spheres=1 → sphere at cube center
        num_spheres=8 → spheres at 8 corners
        num_spheres=9 → 8 corners + 1 center
        otherwise     → uniform grid, random subset of num_spheres points

    Adds node sets "left" (x≈0) and "right" (x≈Lx).
    Saves both .msh and .vtk in case_dir.
    """

    random.seed(seed)
    gmsh.initialize()
    gmsh.model.add("cube_with_spheres")

    # --- 1. Cube geometry
    box = gmsh.model.occ.addBox(0, 0, 0, Lx, Ly, Lz)

    # --- 2. Sphere placement logic
    r = (min_radius + max_radius) / 2.0
    centers = []

    if num_spheres == 1:
        centers = [(Lx/2, Ly/2, Lz/2)]

    elif num_spheres == 8:
        centers = [(x, y, z)
                   for x in [0, Lx]
                   for y in [0, Ly]
                   for z in [0, Lz]]

    elif num_spheres == 9:
        centers = [(x, y, z)
                   for x in [0, Lx]
                   for y in [0, Ly]
                   for z in [0, Lz]]
        centers.append((Lx/2, Ly/2, Lz/2))

    else:
        n = math.ceil(num_spheres ** (1/3))
        spacing_x = Lx / (n - 1 if n > 1 else 1)
        spacing_y = Ly / (n - 1 if n > 1 else 1)
        spacing_z = Lz / (n - 1 if n > 1 else 1)

        all_points = [(i*spacing_x, j*spacing_y, k*spacing_z)
                      for i in range(n)
                      for j in range(n)
                      for k in range(n)]
        centers = random.sample(all_points, min(num_spheres, len(all_points)))

    # --- 3. Add spheres + cut
    sphere_tags = [gmsh.model.occ.addSphere(x, y, z, r) for (x, y, z) in centers]
    gmsh.model.occ.cut([(3, box)], [(3, s) for s in sphere_tags],
                       removeObject=True, removeTool=True)
    gmsh.model.occ.synchronize()

    # --- 4. Mesh (tetrahedra)
    if mesh_size_min is not None:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size_min)
    else:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", r * 0.5)

    if mesh_size_max is not None:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_max)
    else:
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", r * 1.0)

    gmsh.model.mesh.generate(3)

    # --- 5. Save to files
    os.makedirs(case_dir, exist_ok=True)
    msh_file = os.path.join(case_dir, "cube_with_spheres.msh")
    vtk_file = os.path.join(case_dir, "cube_with_spheres.vtk")
    gmsh.write(msh_file)
    gmsh.write(vtk_file)
    gmsh.finalize()

    # --- 6. Wrap in FOL Mesh object
    mesh = meshio.read(msh_file)
    fe_mesh = Mesh("cube_with_spheres_io", "cube_with_spheres.")
    fe_mesh.node_ids = jnp.arange(len(mesh.points))
    fe_mesh.nodes_coordinates = jnp.array(mesh.points)

    if "tetra" not in mesh.cells_dict:
        raise RuntimeError("No tetrahedral cells found. Check meshing settings.")
    fe_mesh.elements_nodes = {"tetra": jnp.array(mesh.cells_dict["tetra"])}
# --- 7. Define boundary node sets (left & right faces)
    tol = 1e-6
    coords = fe_mesh.nodes_coordinates
    left_ids  = fe_mesh.node_ids[jnp.isclose(coords[:,0], 0.0, atol=tol)]
    right_ids = fe_mesh.node_ids[jnp.isclose(coords[:,0], Lx, atol=tol)]
    fe_mesh.node_sets = {"left": left_ids, "right": right_ids}

    # rebuild clean meshio object to avoid bad cell_sets
    fe_mesh.mesh_io = meshio.Mesh(
        points=jnp.array(fe_mesh.nodes_coordinates),
        cells={"tetra": jnp.array(fe_mesh.elements_nodes["tetra"])}
    )

    fe_mesh.is_initialized = True
    return fe_mesh


def create_hex_mesh_with_spheres(Lx, Ly, Lz, nx, ny, nz, sphere_centers, sphere_radii, case_dir):
    """
    Create a structured hexahedral mesh of a box with spherical voids (voxelization style).
    
    Parameters
    ----------
    Lx, Ly, Lz : float
        Box dimensions
    nx, ny, nz : int
        Number of divisions in each direction
    sphere_centers : list of (x,y,z)
        Centers of spheres
    sphere_radii : list of float
        Radii of spheres
    case_dir : str
        Output directory
    """

    os.makedirs(case_dir, exist_ok=True)

    # Grid points
    x = np.linspace(0, Lx, nx+1)
    y = np.linspace(0, Ly, ny+1)
    z = np.linspace(0, Lz, nz+1)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Hexahedral connectivity
    hexes = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # indices of cube corners
                n0 = i*(ny+1)*(nz+1) + j*(nz+1) + k
                n1 = n0 + 1
                n2 = n0 + (nz+1)
                n3 = n2 + 1
                n4 = n0 + (ny+1)*(nz+1)
                n5 = n4 + 1
                n6 = n4 + (nz+1)
                n7 = n6 + 1
                hexes.append([n0,n1,n3,n2,n4,n5,n7,n6])
    hexes = np.array(hexes)

    # Remove hexes inside spheres
    hex_centers = points[hexes].mean(axis=1)
    mask = np.ones(len(hexes), dtype=bool)
    for c, r in zip(sphere_centers, sphere_radii):
        dist = np.linalg.norm(hex_centers - np.array(c), axis=1)
        mask &= dist > r  # keep only outside
    hexes = hexes[mask]

    # Write mesh
    mesh = meshio.Mesh(points, [("hexahedron", hexes)])
    msh_file = os.path.join(case_dir, "cube_with_spheres_hex.msh")
    vtk_file = os.path.join(case_dir, "cube_with_spheres_hex.vtk")
    meshio.write(msh_file, mesh)
    meshio.write(vtk_file, mesh)

    return mesh

def plot_norm_iter(data,plot_name='res_norm_iter',type=None):

    
    if type=='1':
        plt.figure(figsize=(10,4))
    else:
        plt.figure(figsize=(2,4))
        
    plt.plot(np.arange(len(data)), data, marker='o', color='black')
    plt.yscale("log")
    plt.grid(which="both", linestyle="--", linewidth=0.7, alpha=0.7)  # clearer grid
    
    plt.xlabel("Iteration",fontdict={"size": 16})
    plt.xlim()
    if type=='1':    
        plt.ylabel("Residual norm",fontdict={"size": 16})
    plt.ylim((1e-9,3e-1))
    
    # # set x-axis ticks every 5
    # plt.xticks(np.arange(0, len(data) + 1, 5))
    
    plt.tight_layout()
    plt.savefig(f"{plot_name}.png")
    print(f"plot saved to {plot_name}")
    plt.close()