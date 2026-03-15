import math
import numpy as np
from conductivity import conductivity_values
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

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

    # Grid parameters (same as before)
    dx = 0.5  # mm
    dy = 0.5  # mm
    dz = 0.1  # mm

    xmin = min(b.start_x for b in all_boxes)
    xmax = max(b.end_x for b in all_boxes)
    ymin = min(b.start_y for b in all_boxes)
    ymax = max(b.end_y for b in all_boxes)
    zmin = min(b.start_z for b in all_boxes)
    zmax = max(b.end_z for b in all_boxes)

    nx = max(1, int(math.ceil((xmax - xmin) / dx)))
    ny = max(1, int(math.ceil((ymax - ymin) / dy)))
    nz = max(1, int(math.ceil((zmax - zmin) / dz)))

    # Build 3D conductivity grid (same as before)
    k_grid = np.zeros((nx, ny, nz), dtype=float)
    power_grid = np.zeros((nx, ny, nz), dtype=float)

    # Populate grids from boxes (your existing logic)
    for box in all_boxes:
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x = xmin + i * dx
                    y = ymin + j * dy
                    z = zmin + k * dz
                    
                    if (box.start_x <= x < box.end_x and
                        box.start_y <= y < box.end_y and
                        box.start_z <= z < box.end_z):
                        k_grid[i, j, k] = conductivity_values.get(box.material, 100)

    if power_dict:
        for box_name, power in power_dict.items():
            for box in all_boxes:
                if box.name == box_name:
                    for i in range(nx):
                        for j in range(ny):
                            for k in range(nz):
                                x = xmin + i * dx
                                y = ymin + j * dy
                                z = zmin + k * dz
                                if (box.start_x <= x < box.end_x and
                                    box.start_y <= y < box.end_y and
                                    box.start_z <= z < box.end_z):
                                    power_grid[i, j, k] = power / (np.sum(power_grid > 0) if np.sum(power_grid > 0) else 1)

    # Convert grid to PySpice netlist
    circuit = Circuit('Thermal Network')
    
    # Create SPICE netlist from RC grid
    node_id = {}
    node_counter = 1
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if k_grid[i, j, k] > 0:
                    node_id[(i, j, k)] = node_counter
                    node_counter += 1

    # Add resistances between adjacent voxels
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if (i, j, k) not in node_id:
                    continue
                
                curr_node = node_id[(i, j, k)]
                k_curr = k_grid[i, j, k]
                
                # Resistance in +X direction
                if i + 1 < nx and (i + 1, j, k) in node_id:
                    k_next = k_grid[i + 1, j, k]
                    # R = distance / (k * area)
                    R_x = dx / (k_curr * dy * dz) if k_curr > 0 else 1e6
                    circuit.R(f'rx_{i}_{j}_{k}', curr_node, node_id[(i + 1, j, k)], f'{R_x}@u_kΩ')
                
                # Resistance in +Y direction
                if j + 1 < ny and (i, j + 1, k) in node_id:
                    k_next = k_grid[i, j + 1, k]
                    R_y = dy / (k_curr * dx * dz) if k_curr > 0 else 1e6
                    circuit.R(f'ry_{i}_{j}_{k}', curr_node, node_id[(i, j + 1, k)], f'{R_y}@u_kΩ')
                
                # Resistance in +Z direction
                if k + 1 < nz and (i, j, k + 1) in node_id:
                    k_next = k_grid[i, j, k + 1]
                    R_z = dz / (k_curr * dx * dy) if k_curr > 0 else 1e6
                    circuit.R(f'rz_{i}_{j}_{k}', curr_node, node_id[(i, j, k + 1)], f'{R_z}@u_kΩ')

    # Add power sources
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if power_grid[i, j, k] > 0 and (i, j, k) in node_id:
                    # Current source (heat) to ground
                    circuit.I(f'heat_{i}_{j}_{k}', node_id[(i, j, k)], circuit.gnd, f'{power_grid[i, j, k]}@u_A')

    # Run PySpice simulation (DC operating point for steady-state)
    try:
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.operating_point()
        
        # Extract temperatures from nodes
        temps = {}
        for (i, j, k), node in node_id.items():
            temps[(i, j, k)] = float(analysis[node])
        
        # Map temperatures back to boxes
        results = {}
        for box in boxes:
            box_temps = []
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        x = xmin + i * dx
                        y = ymin + j * dy
                        z = zmin + k * dz
                        if (box.start_x <= x < box.end_x and
                            box.start_y <= y < box.end_y and
                            box.start_z <= z < box.end_z and
                            (i, j, k) in temps):
                            box_temps.append(temps[(i, j, k)])
            
            if box_temps:
                peak_temp = max(box_temps)
                avg_temp = np.mean(box_temps)
                results[box.name] = (peak_temp, avg_temp, 0, 0, 0)
        
        return results
    
    except Exception as e:
        print(f"PySpice simulation error: {e}")
        return {}
