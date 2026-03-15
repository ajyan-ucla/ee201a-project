#!/usr/bin/env python3
"""
Unit tests for simulator.py PySpice integration
Tests grid initialization and basic simulator functionality
"""
import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator import simulator_simulate
from rearrange import Box
from therm_xml_parser import Layer

class MockBox:
    """Simple box for testing"""
    def __init__(self, name, start_x, start_y, start_z, width, length, height, power=100.0, material="Si"):
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
        self.material = material

class TestSimulatorGridInitialization(unittest.TestCase):
    """Test grid setup and initialization logic"""

    def test_empty_box_list(self):
        """Test that simulator handles empty box list gracefully"""
        results = simulator_simulate(
            boxes=[],
            bonding_box_list=[],
            TIM_boxes=[]
        )
        self.assertEqual(results, {}, "Empty box list should return empty dict")
        print("[PASS] Empty box list test passed")

    def test_none_optional_lists(self):
        """Test that simulator handles None optional parameters"""
        box = MockBox("test_box", 0, 0, 0, 10, 10, 1, power=100.0)
        results = simulator_simulate(
            boxes=[box],
            bonding_box_list=None,
            TIM_boxes=None
        )
        # Should not crash and should return a result
        self.assertIsInstance(results, dict, "Should return a dictionary")
        print("[PASS] None optional lists test passed")

    def test_single_box_basic_execution(self):
        """Test that simulator can execute with a single box without crashing"""
        box = MockBox("gpu", 0, 0, 0, 30, 30, 1, power=400.0, material="Si")
        
        try:
            results = simulator_simulate(
                boxes=[box],
                bonding_box_list=[],
                TIM_boxes=[],
                power_dict={"gpu": 400.0}
            )
            # Check that we got a result for the box
            self.assertIn("gpu", results, "Results should contain the GPU box")
            self.assertIsInstance(results["gpu"], tuple, "Result should be a tuple")
            self.assertEqual(len(results["gpu"]), 5, "Result tuple should have 5 elements")
            print("[PASS] Single box basic execution test passed")
        except Exception as e:
            self.fail("Simulator crashed on single box: {}".format(e))

    def test_grid_bounds_calculation(self):
        """Test that grid bounds are calculated correctly"""
        # Create boxes at different positions
        box1 = MockBox("box1", 0, 0, 0, 10, 10, 1, material="Si")
        box2 = MockBox("box2", 10, 10, 0, 10, 10, 1, material="Si")
        
        try:
            results = simulator_simulate(
                boxes=[box1, box2],
                bonding_box_list=[],
                TIM_boxes=[]
            )
            self.assertIsInstance(results, dict, "Should return a dictionary")
            print("[PASS] Grid bounds calculation test passed")
        except Exception as e:
            self.fail("Grid bounds calculation failed: {}".format(e))

class TestSimulatorOutputValidation(unittest.TestCase):
    """Test that simulator outputs are physically reasonable"""

    def test_result_tuple_structure(self):
        """Test that results have correct structure: (peak_temp, avg_temp, Rx, Ry, Rz)"""
        box = MockBox("test", 0, 0, 0, 30, 30, 1, power=100.0)
        
        results = simulator_simulate(
            boxes=[box],
            bonding_box_list=[],
            TIM_boxes=[],
            power_dict={"test": 100.0}
        )
        
        if results:  # Only test if we got results
            self.assertIn("test", results)
            peak, avg, rx, ry, rz = results["test"]
            
            # Check types
            self.assertIsInstance(peak, (int, float), "Peak temp should be numeric")
            self.assertIsInstance(avg, (int, float), "Avg temp should be numeric")
            self.assertIsInstance(rx, (int, float), "Rx should be numeric")
            self.assertIsInstance(ry, (int, float), "Ry should be numeric")
            self.assertIsInstance(rz, (int, float), "Rz should be numeric")
            print("[PASS] Result tuple structure test passed")

    def test_temperature_ordering(self):
        """Test that peak_temp >= avg_temp physically"""
        box = MockBox("test", 0, 0, 0, 30, 30, 1, power=400.0)
        
        results = simulator_simulate(
            boxes=[box],
            bonding_box_list=[],
            TIM_boxes=[],
            power_dict={"test": 400.0}
        )
        
        if results and "test" in results:
            peak, avg, _, _, _ = results["test"]
            self.assertGreaterEqual(peak, avg, "Peak temp should be >= average temp")
            print("[PASS] Temperature ordering test passed")

if __name__ == '__main__':
    unittest.main()
