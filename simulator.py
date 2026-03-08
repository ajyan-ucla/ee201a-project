import math
import numpy as np


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

    # Input validation
    if not boxes or len(boxes) == 0:
        print("Warning: No boxes provided to simulator")
        return {}
    
    if not bonding_box_list:
        bonding_box_list = []
    if not TIM_boxes:
        TIM_boxes = []
    
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
    dx = 0.5  # mm  <-- coarse starting point; refine later
    dy = 0.5  # mm
    dz = 0.1  # mm  <-- often smaller in z is helpful

    # ------------------------------------------------------------------
    # 2) Determine global bounding box
    # ------------------------------------------------------------------
    xmin = min(b.start_x for b in all_boxes)
    xmax = max(b.end_x   for b in all_boxes)
    ymin = min(b.start_y for b in all_boxes)
    ymax = max(b.end_y   for b in all_boxes)
    zmin = min(b.start_z for b in all_boxes)
    zmax = max(b.end_z   for b in all_boxes)

    nx = max(1, int(math.ceil((xmax - xmin) / dx)))
    ny = max(1, int(math.ceil((ymax - ymin) / dy)))
    nz = max(1, int(math.ceil((zmax - zmin) / dz)))

    # ------------------------------------------------------------------
    # 3) Allocate grid property arrays
    # ------------------------------------------------------------------
    # Thermal conductivity per voxel
    k_grid = np.zeros((nx, ny, nz), dtype=float)

    # Power injected into each voxel [W]
    p_grid = np.zeros((nx, ny, nz), dtype=float)

    # Box owner index / debugging aid
    owner_grid = np.full((nx, ny, nz), fill_value=-1, dtype=int)

    # Ambient/default values
    k_air = 0.026  # W/(m*K), if you ever leave empty space in the model
    T_amb = 25.0   # degC

    # ------------------------------------------------------------------
    # 4) Fill material / conductivity grid
    # ------------------------------------------------------------------
    # Strategy:
    # For each voxel center, figure out which box contains it, then assign k.
    # Start simple: one effective isotropic k per box.
    # Later you can improve stackup handling.
    # ------------------------------------------------------------------
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x, y, z = voxel_center(i, j, k, xmin, ymin, zmin, dx, dy, dz)

                box_idx = find_containing_box_index(all_boxes, x, y, z)
                if box_idx is None:
                    # Outside all solids
                    k_grid[i, j, k] = k_air
                    continue

                box = all_boxes[box_idx]
                owner_grid[i, j, k] = box_idx
                k_grid[i, j, k] = effective_box_conductivity(box, layers)

    # ------------------------------------------------------------------
    # 5) Fill power grid
    # ------------------------------------------------------------------
    # Distribute each powered box's total power uniformly over its voxels.
    # PDF says use GPU = 400 W and each HBM = 5 W. therm.py may already
    # populate box.power, so prefer box.power when available. :contentReference[oaicite:2]{index=2}
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
    # Simplest physically reasonable version:
    # connect top voxels to ambient through convection.
    # g_conv = h * A
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
    # T is temperature rise or absolute temperature depending on your setup.
    # Here, because ambient is stamped directly into b, T solves to absolute degC.
    # ------------------------------------------------------------------
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
    Extract effective thermal conductivity from box stackup.
    
    Stackup format: "N:layer_name,M:layer_name2"
    Example: "1:5nm_active,2:5nm_advanced_metal"
    
    Returns effective k considering all layers in series.
    """
    if layers is None or not hasattr(box, 'stackup') or not box.stackup:
        # Fallback: guess from name
        name_lower = box.name.lower()
        if "gpu" in name_lower or "hbm" in name_lower:
            return 105.0  # Si
        elif "tim" in name_lower:
            return 100.0  # TIM
        elif "bond" in name_lower:
            return 36.0   # SnPb solder
        else:
            return 10.0   # generic
    
    # Import conductivity dictionary
    from therm import conductivity_values
    
    # Parse stackup string: split by comma
    try:
        stackup_specs = str(box.stackup).split(",")
    except:
        return 10.0
    
    # Collect all layer names with their thicknesses
    total_resistance = 0.0  # K / W (for 1 mm^2 cross-section)
    total_thickness = 0.0   # mm
    
    for spec in stackup_specs:
        spec = spec.strip()
        if ":" not in spec:
            continue
        
        try:
            # Parse "N:layer_name"
            num_str, layer_name = spec.split(":")
            num_layers = int(num_str)
            layer_name = layer_name.strip()
        except:
            continue
        
        # Find layer in layers list
        layer_obj = None
        if isinstance(layers, list):
            for layer in layers:
                if hasattr(layer, 'get_name') and layer.get_name() == layer_name:
                    layer_obj = layer
                    break
                elif hasattr(layer, 'name') and layer.name == layer_name:
                    layer_obj = layer
                    break
        elif isinstance(layers, dict):
            layer_obj = layers.get(layer_name)
        
        if layer_obj is None:
            continue
        
        # Get layer properties
        if hasattr(layer_obj, 'get_thickness'):
            thickness = layer_obj.get_thickness()
        elif hasattr(layer_obj, 'thickness'):
            thickness = layer_obj.thickness
        else:
            thickness = 0.1  # mm, fallback
        
        if hasattr(layer_obj, 'get_material'):
            material = layer_obj.get_material()
        elif hasattr(layer_obj, 'material'):
            material = layer_obj.material
        else:
            material = layer_name
        
        # Get conductivity for this material
        k = conductivity_values.get(material, 10.0)  # W/(m·K)
        
        # Add resistance contribution from this layer
        # R = thickness / (k * A), but we track per unit area
        # Total thickness * num_layers
        layer_total_thickness = thickness * num_layers
        layer_resistance = layer_total_thickness / k  # relative resistance
        
        total_resistance += layer_resistance
        total_thickness += layer_total_thickness
    
    if total_thickness <= 0 or total_resistance <= 0:
        return 10.0
    
    # Effective k = total_thickness / total_resistance
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
