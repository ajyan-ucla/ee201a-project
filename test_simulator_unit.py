#!/usr/bin/env python3
"""
Unit tests for simulator.py functions
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator import (
    voxel_center, node_id, point_in_box, 
    effective_box_conductivity, assign_power_to_grid
)
from rearrange import Box
from therm_xml_parser import Layer

class MockBox:
    """Simple box for testing"""
    def __init__(self, name, start_x, start_y, start_z, width, length, height, power=100.0, stackup="1:Si"):
        self.name = name
        self.start_x = start_x
        self.start_y = start_y
        self.start_z = start_z
        self.width = width
        self.length = length
        self.height = height
        self.end_x = start_x + width
        self.end_y = start_y + length
        self.end_z = start_z + height
        self.power = power
        self.stackup = stackup

def test_voxel_center():
    """Test voxel center calculation"""
    print("Testing voxel_center...")
    x, y, z = voxel_center(0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0)
    assert x == 0.5 and y == 0.5 and z == 0.5, f"Expected (0.5, 0.5, 0.5), got ({x}, {y}, {z})"
    print("  PASS voxel_center passed")

def test_node_id():
    """Test node ID calculation"""
    print("Testing node_id...")
    # For nx=10, ny=10, nz=10:
    # node at (0,0,0) should be 0
    # node at (1,0,0) should be 1
    # node at (0,1,0) should be 10
    idx = node_id(0, 0, 0, 10, 10, 10)
    assert idx == 0, f"Expected 0, got {idx}"
    
    idx = node_id(1, 0, 0, 10, 10, 10)
    assert idx == 1, f"Expected 1, got {idx}"
    
    idx = node_id(0, 1, 0, 10, 10, 10)
    assert idx == 10, f"Expected 10, got {idx}"
    
    print("  PASS node_id passed")

def test_point_in_box():
    """Test point containment"""
    print("Testing point_in_box...")
    box = MockBox("test", 0, 0, 0, 10, 10, 10)
    
    assert point_in_box(box, 5, 5, 5) == True, "Point inside should return True"
    assert point_in_box(box, -1, 5, 5) == False, "Point outside should return False"
    assert point_in_box(box, 10, 5, 5) == False, "Point on boundary should return False"
    
    print("  PASS point_in_box passed")

def test_effective_box_conductivity():
    """Test conductivity lookup"""
    print("Testing effective_box_conductivity...")
    
    si_layer = Layer(
        name="Si",
        active=True,
        cost_per_mm2=0.1,
        defect_density=0.01,
        critical_area_ratio=0.5,
        clustering_factor=1.0,
        litho_percent=0.0,
        mask_cost=0.0,
        stitching_yield=1.0,
        static=True,
        thickness=1.0,
        material="Si"
    )
    layers = [si_layer]
    
    box = MockBox("gpu", 0, 0, 0, 10, 10, 10, stackup="1:Si")
    k = effective_box_conductivity(box, layers)
    assert k > 100, f"Si conductivity should be high (~105), got {k}"
    
    print("  PASS effective_box_conductivity passed")

def test_assign_power_to_grid():
    """Test power distribution"""
    print("Testing assign_power_to_grid...")
    
    boxes = [
        MockBox("box1", 0, 0, 0, 2, 2, 2, power=100.0),
        MockBox("box2", 2, 0, 0, 2, 2, 2, power=200.0),
    ]
    
    # Create owner grid explicitly - 4 voxels for box1, 4 voxels for box2
    owner_grid = np.zeros((2, 2, 2), dtype=int)
    owner_grid[0, :, :] = 0  # First half is box1
    owner_grid[1, :, :] = 1  # Second half is box2
    
    p_grid = np.zeros((2, 2, 2))
    
    assign_power_to_grid(boxes, owner_grid, p_grid)
    
    # Box1: 100W / 4 voxels = 25W per voxel
    # Box2: 200W / 4 voxels = 50W per voxel
    assert np.allclose(p_grid[0, 0, 0], 100.0 / 4), f"Box1 power distribution wrong"
    assert np.allclose(p_grid[1, 0, 0], 200.0 / 4), f"Box2 power distribution wrong"
    
    print("  PASS assign_power_to_grid passed")

if __name__ == "__main__":
    print("=" * 50)
    print("UNIT TESTS FOR SIMULATOR")
    print("=" * 50)
    
    try:
        test_voxel_center()
        test_node_id()
        test_point_in_box()
        test_effective_box_conductivity()
        test_assign_power_to_grid()
        
        print("\n" + "=" * 50)
        print("ALL UNIT TESTS PASSED")
        print("=" * 50)
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
