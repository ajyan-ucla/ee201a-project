import math
import numpy as np
from conductivity import conductivity_values
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

def effective_box_conductivity(box, layers):
    """
    Handle both simple and composite stackups.
    
    Simple: "1:layer_name,2:layer_name2"
    Composite: "1:material1:50,material2:50"
    """
    if layers is None or not hasattr(box, 'stackup') or not box.stackup:
        # Fallback logic
        ...
    
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
    3D grid-based thermal RC solver using PySpice.
    Returns: results[box.name] = (peak_temp, avg_temp, Rx, Ry, Rz)
    """
    
    if not boxes or len(boxes) == 0:
        return {}

    if not bonding_box_list:
        bonding_box_list = []
    if not TIM_boxes:
        TIM_boxes = []

    all_boxes = list(boxes) + list(bonding_box_list) + list(TIM_boxes)
    if len(all_boxes) == 0:
        return {}

    # Grid parameters
    dx = 0.5  # mm
    dy = 0.5  # mm
    dz = 0.1  # mm

    # Compute bounding box
    xmin = min(b.start_x for b in all_boxes)
    xmax = max(b.end_x for b in all_boxes)
    ymin = min(b.start_y for b in all_boxes)
    ymax = max(b.end_y for b in all_boxes)
    zmin = min(b.start_z for b in all_boxes)
    zmax = max(b.end_z for b in all_boxes)

    # Computer the number of voxels that fit in each of the bounding box dimensions
    nx = max(1, int(math.ceil((xmax - xmin) / dx)))
    ny = max(1, int(math.ceil((ymax - ymin) / dy)))
    nz = max(1, int(math.ceil((zmax - zmin) / dz)))

    # Initialize conductivity, power dissipation, and owner grids
    k_grid = np.zeros((nx, ny, nz), dtype=float)                    # Stores thermal conductivity at each voxel
    power_grid = np.zeros((nx, ny, nz), dtype=float)                # Stores heat generation at each voxel                              
    owner_grid = np.full((nx, ny, nz), fill_value=-1, dtype=int)    # Maps which box each voxel belongs to

    # Create coordinate grids
    i_indices = np.arange(nx)
    j_indices = np.arange(ny)
    k_indices = np.arange(nz)

    i_grid, j_grid, k_grid_idx = np.meshgrid(i_indices, j_indices, k_indices, indexing='ij')
    
    x_grid = xmin + (i_grid + 0.5) * dx
    y_grid = ymin + (j_grid + 0.5) * dy
    z_grid = zmin + (k_grid_idx + 0.5) * dz
    
    # Build owner grid
    for box_idx, box in enumerate(all_boxes):
        mask = ((box.start_x <= x_grid) & (x_grid < box.end_x) &
                (box.start_y <= y_grid) & (y_grid < box.end_y) &
                (box.start_z <= z_grid) & (z_grid < box.end_z))
        owner_grid[mask] = box_idx
    
    # Fill k_grid
    k_grid[:, :, :] = 0.026 # Initialize to air
    for box_idx, box in enumerate(all_boxes):
        mask = owner_grid == box_idx
        k_grid[mask] = effective_box_conductivity(box, layers)
    
    # Fill power_grid
    if power_dict:
        voxel_counts = np.zeros(len(all_boxes), dtype=int)
        for box_idx in range(len(all_boxes)):
            voxel_counts[box_idx] = np.sum(owner_grid == box_idx)
        
        for box_idx, box in enumerate(all_boxes):
            if box.name in power_dict and voxel_counts[box_idx] > 0:
                mask = owner_grid == box_idx
                power_grid[mask] = power_dict[box.name] / voxel_counts[box_idx]

    circuit = Circuit('Thermal Network')
    
    node_id = {}
    node_counter = 1

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if k_grid[i, j, k] > 0:
                    node_id[(i, j, k)] = node_counter
                    node_counter += 1
    
    dx_m = dx * 1e-3
    dy_m = dy * 1e-3
    dz_m = dz * 1e-3
    
    heatsink_r = 0.01
    ambient_temp = 45
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if (i, j, k) not in node_id:
                    continue
                
                curr_node = node_id[(i, j, k)]
                k_curr = k_grid[i, j, k]
                
                if i + 1 < nx and (i + 1, j, k) in node_id:
                    k_next = k_grid[i + 1, j, k]
                    k_avg = (k_curr + k_next) / 2
                    R_x = dx_m / (k_avg * dy_m * dz_m)
                    circuit.R(f'rx_{i}_{j}_{k}', curr_node, node_id[(i + 1, j, k)], f'{R_x}@u_Ohm')
                
                if j + 1 < ny and (i, j + 1, k) in node_id:
                    k_next = k_grid[i, j + 1, k]
                    k_avg = (k_curr + k_next) / 2
                    R_y = dy_m / (k_avg * dx_m * dz_m)
                    circuit.R(f'ry_{i}_{j}_{k}', curr_node, node_id[(i, j + 1, k)], f'{R_y}@u_Ohm')
                
                if k + 1 < nz and (i, j, k + 1) in node_id:
                    k_next = k_grid[i, j, k + 1]
                    k_avg = (k_curr + k_next) / 2
                    R_z = dz_m / (k_avg * dx_m * dy_m)
                    circuit.R(f'rz_{i}_{j}_{k}', curr_node, node_id[(i, j, k + 1)], f'{R_z}@u_Ohm')
                else:
                    if k == nz - 1:
                        circuit.R(f'rh_{i}_{j}_{k}', curr_node, circuit.gnd, f'{heatsink_r}@u_Ohm')
                
                if power_grid[i, j, k] > 0:
                    circuit.I(f'heat_{i}_{j}_{k}', curr_node, circuit.gnd, f'{power_grid[i, j, k]}@u_A')   

    # Run PySpice simulation (DC operating point for steady-state)
    try:
        simulator = circuit.simulator(temperature=ambient_temp, nominal_temperature=ambient_temp)
        analysis = simulator.operating_point()
        
        temps = {}
        for (i, j, k), node in node_id.items():
            temps[(i, j, k)] = float(analysis[node])
        
        results = {}
        for box_idx, box in enumerate(all_boxes):
            box_temps = []
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        if owner_grid[i, j, k] == box_idx and (i, j, k) in temps:
                            box_temps.append(temps[(i, j, k)])
            
            if box_temps:
                peak_temp = max(box_temps)
                avg_temp = np.mean(box_temps)
                
                k_box = effective_box_conductivity(box, layers)

                # Thermal resistance calculation based on geometry and material conductivity
                Rx = (box.width * 1e-3) / (k_box * box.length * 1e-3 * box.height * 1e-3)
                Ry = (box.length * 1e-3) / (k_box * box.width * 1e-3 * box.height * 1e-3)
                Rz = (box.height * 1e-3) / (k_box * box.width * 1e-3 * box.length * 1e-3)
                
                results[box.name] = (peak_temp, avg_temp, Rx, Ry, Rz)
        
        return results
    
    except Exception as e:
        print(f"PySpice simulation error: {e}")
        return {}
