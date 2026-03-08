```python
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
    print("  ✓ voxel_center passed")

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
    
    print("  ✓ node_id passed")

def test_point_in_box():
    """Test point containment"""
    print("Testing point_in_box...")
    box = MockBox("test", 0, 0, 0, 10, 10, 10)
    
    assert point_in_box(box, 5, 5, 5) == True, "Point inside should return True"
    assert point_in_box(box, -1, 5, 5) == False, "Point outside should return False"
    assert point_in_box(box, 10, 5, 5) == False, "Point on boundary should return False"
    
    print("  ✓ point_in_box passed")

def test_effective_box_conductivity():
    """Test conductivity lookup"""
    print("Testing effective_box_conductivity...")
    box = MockBox("gpu", 0, 0, 0, 10, 10, 10, stackup="1:Si")
    
    # Test with no layers (should use fallback)
    k = effective_box_conductivity(box, None)
    assert k > 0, f"Conductivity should be positive, got {k}"
    
    # Test with GPU in name
    box_gpu = MockBox("GPU", 0, 0, 0, 10, 10, 10)
    k_gpu = effective_box_conductivity(box_gpu, None)
    assert k_gpu > 100, f"GPU k should be high (~105), got {k_gpu}"
    
    print("  ✓ effective_box_conductivity passed")

def test_assign_power_to_grid():
    """Test power distribution"""
    print("Testing assign_power_to_grid...")
    
    # Create simple boxes
    boxes = [
        MockBox("box1", 0, 0, 0, 2, 2, 2, power=100.0),
        MockBox("box2", 2, 0, 0, 2, 2, 2, power=200.0),
    ]
    
    # Create owner grid (2x2x2 each)
    owner_grid = np.array([
        [[0, 0], [0, 0]],
        [[1, 1], [1, 1]],
    ])
    owner_grid = owner_grid.reshape(2, 2, 2)
    
    p_grid = np.zeros((2, 2, 2))
    
    # Assign power
    assign_power_to_grid(boxes, owner_grid, p_grid)
    
    # Check that box1 got 100W distributed over 8 voxels = 12.5W each
    assert p_grid[0, 0, 0] == 100.0 / 8, f"Box1 power distribution wrong"
    assert p_grid[1, 0, 0] == 200.0 / 8, f"Box2 power distribution wrong"
    
    print("  ✓ assign_power_to_grid passed")

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
        print("✓ ALL UNIT TESTS PASSED")
        print("=" * 50)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
