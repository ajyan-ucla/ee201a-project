import math
import numpy as np
from therm import conductivity_values



def simulator_simulate(
    boxes,
    bonding_box_list,
    TIM_boxes,
    heatsink_obj=None,
    heatsink_list=None,
    heatsink_name=None,
    bonding_list=None,
    bonding_name_type_dict=None,
    is_repeat=False,
    min_TIM_height=0.01,
    power_dict=None,
    anemoi_parameter_ID=None,
    layers=None,
):
    """
    Grid-based steady-state thermal RC solver.

    Expected return:
        results[box.name] = (peak_temp, avg_temp, Rx, Ry, Rz)

    Notes:
    - Start simple: uniform grid, isotropic conductivity, steady-state only.
    - Keep units consistent. The repo appears to use mm for geometry.
    """

    # Input validation - Check if we have any boxes to simulate
    if not boxes or len(boxes) == 0:
        print("Warning: No boxes provided to simulator")
        return {}

    # Initialize empty lists if not provided
    if not bonding_box_list:
        bonding_box_list = []
    if not TIM_boxes:
        TIM_boxes = []
        
    # Combine all thermal solids into one list for processing
    all_boxes = list(boxes) + list(bonding_box_list) + list(TIM_boxes)
    
    if len(all_boxes) == 0:
        print("Warning: No thermal boxes found")
        return {}

    # ------------------------------------------------------------------
    # 0) Collect all thermal solids that should appear in the grid
    # ------------------------------------------------------------------
    all_boxes = list(boxes) + list(bonding_box_list) + list(TIM_boxes)

    # Optional: include heatsink base as another conducting region later
    # if heatsink_obj is not None:
    #     ...

    # ------------------------------------------------------------------
    # 1) Choose grid resolution
    # ------------------------------------------------------------------
    # Define voxel spacing in each direction
    dx = 0.5  # mm  <-- coarse starting point; refine later
    dy = 0.5  # mm
    dz = 0.1  # mm  <-- often smaller in z is helpful

    # ------------------------------------------------------------------
    # 2) Determine global bounding box
    # ------------------------------------------------------------------
    # Find the min/max coordinates of the entire geometry
    xmin = min(b.start_x for b in all_boxes)
    xmax = max(b.end_x   for b in all_boxes)
    ymin = min(b.start_y for b in all_boxes)
    ymax = max(b.end_y   for b in all_boxes)
    zmin = min(b.start_z for b in all_boxes)
    zmax = max(b.end_z   for b in all_boxes)

    # Calculate number of voxels in each direction
    nx = max(1, int(math.ceil((xmax - xmin) / dx)))
    ny = max(1, int(math.ceil((ymax - ymin) / dy)))
    nz = max(1, int(math.ceil((zmax - zmin) / dz)))

    # ------------------------------------------------------------------
    # 3) Allocate grid property arrays
    # ------------------------------------------------------------------
    # Create 3D nump arrays to store properties at each voxel
    
    # THERMAL CONDUCTIVITY GRID
    # k_grid[i,j,k] = thermal conductivity of voxel at grid position (i,j,k)
    # Units: W/(m·K)
    # Initialized to zero; will be filled based on material at that location
    k_grid = np.zeros((nx, ny, nz), dtype=float)

    # POWER GRID
    # p_grid[i,j,k] = power dissipated in voxel (i,j,k)
    # Units: W (total watts, not watts per volume)
    # This is the heat source term in the Fourier equation
    p_grid = np.zeros((nx, ny, nz), dtype=float)

    # OWNER GRID (debugging/bookkeeping)
    # owner_grid[i,j,k] = index of the box that owns/contains this voxel
    # -1 means voxel is outside all solid objects (filled with air/vacuum)
    owner_grid = np.full((nx, ny, nz), fill_value=-1, dtype=int)

    # AMBIENT CONDITIONS
    # k_air: Thermal conductivity of air/void space
    #        If a voxel is not inside any solid box, use air conductivity
    #        (typically you'd set this to 0 if voids are truly empty, but
    #         leaving as low air value prevents singularities)
    k_air = 0.026  # W/(m*K), if you ever leave empty space in the model


    # T_amb: Ambient temperature for boundary conditions
    #        This is the temperature assumed at cooling surfaces (e.g., heatsink)
    T_amb = 25.0   # degC

    # ------------------------------------------------------------------
    # 4) Fill material / conductivity grid
    # ------------------------------------------------------------------
    # Strategy:
    # For each voxel center, figure out which box contains it, then assign k.
    # PHYSICS: We need to know the thermal conductivity at each point in space
    # to build the Laplacian operator. Conductivity can vary by material layers.
    # ------------------------------------------------------------------
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Find the physical (x,y,z) coordinates of the voxel center
                x, y, z = voxel_center(i, j, k, xmin, ymin, zmin, dx, dy, dz)

                # Determine which box (if any) contains this voxel center
                box_idx = find_containing_box_index(all_boxes, x, y, z)
                if box_idx is None:
                    # Outside all solids
                    k_grid[i, j, k] = k_air
                    continue

                # Mark which box owns this voxel (for later postprocessing)
                box = all_boxes[box_idx]
                owner_grid[i, j, k] = box_idx

                # Compute effective conductivity for this box's material/stackup
                # (handles complex multilayer structures)
                k_grid[i, j, k] = effective_box_conductivity(box, layers)

    # ------------------------------------------------------------------
    # 5) Fill power grid
    # ------------------------------------------------------------------
    # Distribute each box's total power uniformly over its voxels
    # 
    # PHYSICS: In the heat equation, P is a source term that adds energy.
    # A box with 400W total dissipation gets that 400W spread uniformly
    # across all voxels inside the box.
    # ------------------------------------------------------------------
    assign_power_to_grid(all_boxes, owner_grid, p_grid)

    # ------------------------------------------------------------------
    # 6) Build linear system G * T = b
    # ------------------------------------------------------------------
    # Dense solve is okay for tiny grids; sparse is better later.
    # Start with dense for simplicity if grid is small enough.
    # ------------------------------------------------------------------
    n_nodes = nx * ny * nz
    G = np.zeros((n_nodes, n_nodes), dtype=float)
    b = np.zeros(n_nodes, dtype=float)

    # Add conduction between neighboring voxels
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                u = node_id(i, j, k, nx, ny, nz)

                # Power injection
                b[u] += p_grid[i, j, k]

                # +x neighbor
                if i + 1 < nx:
                    v = node_id(i + 1, j, k, nx, ny, nz)
                    g = neighbor_conductance_x(k_grid[i, j, k], k_grid[i + 1, j, k], dy, dz, dx)
                    stamp_conductance(G, u, v, g)

                # +y neighbor
                if j + 1 < ny:
                    v = node_id(i, j + 1, k, nx, ny, nz)
                    g = neighbor_conductance_y(k_grid[i, j, k], k_grid[i, j + 1, k], dx, dz, dy)
                    stamp_conductance(G, u, v, g)

                # +z neighbor
                if k + 1 < nz:
                    v = node_id(i, j, k + 1, nx, ny, nz)
                    g = neighbor_conductance_z(k_grid[i, j, k], k_grid[i, j, k + 1], dx, dy, dz)
                    stamp_conductance(G, u, v, g)

    # ------------------------------------------------------------------
    # 7) Apply top-surface cooling boundary
    # ------------------------------------------------------------------
    # Model convective heat transfer from the top surface to the ambient.
    # PHYSICS: Convection is modeled as:
    #          Q = h * A * (T_surface - T_ambient)
    #   where h is the heat transfer coefficient (W/m²·K)
    #         A is the surface area (m²)
    # 
    # In thermal network form: G_conv = h * A
    # This is stamped into the diagonal as:
    #          G[u,u] += G_conv
    #          b[u] += G_conv * T_amb
    # which enforces: G_conv*(T_u - T_amb) term in the balance.
    # Need unit consistency if h comes in W/(m^2*K) while dx,dy are in mm.
    # ------------------------------------------------------------------
    h = extract_htc_from_heatsink(heatsink_obj, default_h=5000.0)

    for i in range(nx):
        for j in range(ny):
            k = nz - 1  # top layer
            u = node_id(i, j, k, nx, ny, nz)

            area_m2 = (dx * 1e-3) * (dy * 1e-3)  # mm^2 -> m^2
            g_conv = h * area_m2

            G[u, u] += g_conv
            b[u] += g_conv * T_amb

    # ------------------------------------------------------------------
    # 8) Solve system
    # ------------------------------------------------------------------
    # G is an (n_nodes × n_nodes) conductance matrix
    # T is the unknown temperature vector (n_nodes × 1)
    # b is the RHS vector (n_nodes × 1) containing power sources
    # 
    # Solution method: Direct linear solve (np.linalg.solve)
    # ------------------------------------------------------------------
    # Check condition number of matrix
    try:
        cond_number = np.linalg.cond(G)
        if cond_number > 1e15:
            print(f"Warning: Matrix condition number is high ({cond_number:.2e}), results may be inaccurate")
    except:
        pass
    
    # Solve the system
    try:
        T = np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        print("Error: Could not solve thermal matrix (singular)")
        T = np.full(n_nodes, T_amb)

    # ------------------------------------------------------------------
    # 9) Postprocess box temperatures
    # ------------------------------------------------------------------
    results = {}
    for box in boxes:
        voxel_ids = voxels_belonging_to_box(box, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz)
        temps = [T[node_id(i, j, k, nx, ny, nz)] for (i, j, k) in voxel_ids]

        if len(temps) == 0:
            peak_temp = T_amb
            avg_temp = T_amb
        else:
            peak_temp = float(np.max(temps))
            avg_temp = float(np.mean(temps))

        # Representative per-box thermal resistances
        Rx, Ry, Rz = representative_box_resistances(box, layers, dx, dy, dz)

        results[box.name] = (peak_temp, avg_temp, Rx, Ry, Rz)

    return results


# ======================================================================
# Helper functions
# ======================================================================

def voxel_center(i, j, k, xmin, ymin, zmin, dx, dy, dz):
    x = xmin + (i + 0.5) * dx
    y = ymin + (j + 0.5) * dy
    z = zmin + (k + 0.5) * dz
    return x, y, z


def node_id(i, j, k, nx, ny, nz):
    return (k * ny + j) * nx + i


def point_in_box(box, x, y, z):
    return (
        box.start_x <= x < box.end_x and
        box.start_y <= y < box.end_y and
        box.start_z <= z < box.end_z
    )


def find_containing_box_index(all_boxes, x, y, z):
    # If boxes overlap, you may need a priority rule later.
    for idx, box in enumerate(all_boxes):
        if point_in_box(box, x, y, z):
            return idx
    return None


def effective_box_conductivity(box, layers):
    """
    Handle both simple and composite stackups.
    
    Simple: "1:layer_name,2:layer_name2"
    Composite: "1:material1:50,material2:50"
    """
    if layers is None or not hasattr(box, 'stackup') or not box.stackup:
        # Fallback logic
        ...
    
    from therm import conductivity_values
    
    try:
        stackup_specs = str(box.stackup).split(",")
    except:
        return 10.0
    
    total_resistance = 0.0
    total_thickness = 0.0
    
    for spec in stackup_specs:
        spec = spec.strip()
        if ":" not in spec:
            continue
        
        parts = spec.split(":")
        
        # ===== HANDLE COMPOSITE MATERIALS =====
        if len(parts) >= 3:  # Composite: N:material1:percent1
            try:
                num_layers = int(parts[0])
                material_name = parts[1]
                percentage = float(parts[2])
            except (ValueError, IndexError):
                continue
            
            # Get conductivity for this material
            k = conductivity_values.get(material_name, 10.0)
            
            # Layer object might exist for this material
            layer_obj = None
            if isinstance(layers, list):
                for layer in layers:
                    if hasattr(layer, 'get_material') and layer.get_material() == material_name:
                        layer_obj = layer
                        break
            
            # Get thickness
            if layer_obj and hasattr(layer_obj, 'get_thickness'):
                thickness = layer_obj.get_thickness()
            else:
                thickness = 0.1  # fallback
            
            # Weighted contribution by percentage
            layer_total_thickness = (thickness * num_layers) * (percentage / 100.0)
            layer_resistance = layer_total_thickness / k
            
            total_resistance += layer_resistance
            total_thickness += layer_total_thickness
        
        # ===== HANDLE SIMPLE LAYERS =====
        elif len(parts) == 2:  # Simple: N:layer_name
            try:
                num_str, layer_name = parts
                num_layers = int(num_str)
                layer_name = layer_name.strip()
            except (ValueError, IndexError):
                continue
            
            # Find layer in layers list
            layer_obj = None
            if isinstance(layers, list):
                for layer in layers:
                    if hasattr(layer, 'get_name') and layer.get_name() == layer_name:
                        layer_obj = layer
                        break
            elif isinstance(layers, dict):
                layer_obj = layers.get(layer_name)
            
            if layer_obj is None:
                continue
            
            # Get properties
            if hasattr(layer_obj, 'get_thickness'):
                thickness = layer_obj.get_thickness()
            elif hasattr(layer_obj, 'thickness'):
                thickness = layer_obj.thickness
            else:
                thickness = 0.1
            
            if hasattr(layer_obj, 'get_material'):
                material = layer_obj.get_material()
            elif hasattr(layer_obj, 'material'):
                material = layer_obj.material
            else:
                material = layer_name
            
            # Get conductivity
            k = conductivity_values.get(material, 10.0)
            
            # Add resistance
            layer_total_thickness = thickness * num_layers
            layer_resistance = layer_total_thickness / k
            
            total_resistance += layer_resistance
            total_thickness += layer_total_thickness
    
    if total_thickness <= 0 or total_resistance <= 0:
        return 10.0
    
    k_eff = total_thickness / total_resistance
    return k_eff


def assign_power_to_grid(all_boxes, owner_grid, p_grid):
    """
    Distribute each box's total power uniformly across its owned voxels.
    """
    unique_box_ids = np.unique(owner_grid)
    for box_idx in unique_box_ids:
        if box_idx < 0:
            continue

        box = all_boxes[box_idx]
        power = getattr(box, "power", 0.0)
        if power is None:
            power = 0.0
        power = float(power)  # Ensure it's a number
        if power == 0.0:
            continue

        mask = (owner_grid == box_idx)
        count = int(np.count_nonzero(mask))
        if count > 0:
            p_grid[mask] += power / count


def stamp_conductance(G, u, v, g):
    G[u, u] += g
    G[v, v] += g
    G[u, v] -= g
    G[v, u] -= g


def neighbor_conductance_x(k1, k2, dy, dz, dx):
    # Convert geometry from mm to m for consistent SI thermal equations
    dx_m = dx * 1e-3
    area_m2 = (dy * 1e-3) * (dz * 1e-3)

    # Two-half-resistor model across material boundary
    R = (dx_m / 2.0) / (k1 * area_m2) + (dx_m / 2.0) / (k2 * area_m2)
    return 1.0 / R


def neighbor_conductance_y(k1, k2, dx, dz, dy):
    dy_m = dy * 1e-3
    area_m2 = (dx * 1e-3) * (dz * 1e-3)
    R = (dy_m / 2.0) / (k1 * area_m2) + (dy_m / 2.0) / (k2 * area_m2)
    return 1.0 / R


def neighbor_conductance_z(k1, k2, dx, dy, dz):
    dz_m = dz * 1e-3
    area_m2 = (dx * 1e-3) * (dy * 1e-3)
    R = (dz_m / 2.0) / (k1 * area_m2) + (dz_m / 2.0) / (k2 * area_m2)
    return 1.0 / R


def extract_htc_from_heatsink(heatsink_obj, default_h=5000.0):
    """
    Try to get heat transfer coefficient from heatsink_obj.
    Fallback to a default.
    """
    if heatsink_obj is None:
        return default_h

    # If dict-like
    if isinstance(heatsink_obj, dict):
        if "hc" in heatsink_obj:
            try:
                return float(heatsink_obj["hc"])
            except Exception:
                pass

    return default_h


def voxels_belonging_to_box(box, xmin, ymin, zmin, dx, dy, dz, nx, ny, nz):
    ids = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x, y, z = voxel_center(i, j, k, xmin, ymin, zmin, dx, dy, dz)
                if point_in_box(box, x, y, z):
                    ids.append((i, j, k))
    return ids


def representative_box_resistances(box, layers, dx, dy, dz):
    """
    Simple placeholder representative resistances for reporting.
    """
    k_eff = effective_box_conductivity(box, layers)

    dx_m = dx * 1e-3
    dy_m = dy * 1e-3
    dz_m = dz * 1e-3

    A_yz = dy_m * dz_m
    A_xz = dx_m * dz_m
    A_xy = dx_m * dy_m

    Rx = dx_m / (k_eff * A_yz)
    Ry = dy_m / (k_eff * A_xz)
    Rz = dz_m / (k_eff * A_xy)

    return Rx, Ry, Rz
