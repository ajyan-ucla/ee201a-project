#!/usr/bin/env python3
"""
Integration test: Single box thermal simulation with PySpice
Tests the complete simulator pipeline with realistic thermal scenarios
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator import simulator_simulate
from rearrange import Box
from therm_xml_parser import Layer

def test_single_box_gpu():
    """Test with single GPU box - 400W power dissipation"""
    print("=" * 60)
    print("TEST 1: Single GPU Box (400W)")
    print("=" * 60)
    
    # Create a single GPU box
    # ~30mm x 30mm x 1mm, 400W power, Silicon
    gpu_box = Box(
        start_x=0.0,
        start_y=0.0,
        start_z=0.0,
        width=30.0,      # 30mm
        length=30.0,     # 30mm
        height=1.0,      # 1mm
        power=400.0,     # 400W
        stackup="1:Si",
        ambient_conduct=0.0,
        name="GPU"
    )
    
    # Create minimal layer for reference
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
    
    # Run simulator
    print("Running PySpice simulator...")
    results = simulator_simulate(
        boxes=[gpu_box],
        bonding_box_list=[],
        TIM_boxes=[],
        heatsink_obj={"hc": 5000.0},
        layers=layers,
        power_dict={"GPU": 400.0}
    )
    
    # Check results
    print("\nResults:")
    if results:
        for name, (peak, avg, rx, ry, rz) in results.items():
            print("  {}:".format(name))
            print("    Peak Temperature: {:.2f}C".format(peak))
            print("    Avg Temperature:  {:.2f}C".format(avg))
            print("    Rx:               {:.6f} K/W".format(rx))
            print("    Ry:               {:.6f} K/W".format(ry))
            print("    Rz:               {:.6f} K/W".format(rz))
            
            # Physical plausibility checks
            try:
                assert peak > 25, "Peak temp should be > 25C (ambient), got {:.2f}C".format(peak)
                assert peak < 200, "Peak temp should be < 200C (unrealistic), got {:.2f}C".format(peak)
                assert avg <= peak, "Average should be <= peak, got avg={:.2f}C, peak={:.2f}C".format(avg, peak)
                print("  [PASS] Results are physically reasonable")
            except AssertionError as e:
                print("  [FAIL] Assertion failed: {}".format(e))
                return False
    else:
        print("   [FAIL] No results returned!")
        return False
    
    print("\n[PASS] TEST 1 PASSED\n")
    return True

def test_single_box_hbm():
    """Test with single HBM box - 5W power dissipation"""
    print("=" * 60)
    print("TEST 2: Single HBM Box (5W)")
    print("=" * 60)
    
    # Create a single HBM box
    # ~10mm x 10mm x 1mm, 5W power, Silicon
    hbm_box = Box(
        start_x=30.0,
        start_y=0.0,
        start_z=0.0,
        width=10.0,      # 10mm
        length=10.0,     # 10mm
        height=1.0,      # 1mm
        power=5.0,       # 5W
        stackup="1:Si",
        ambient_conduct=0.0,
        name="HBM"
    )
    
    # Create minimal layer
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
    
    # Run simulator
    print("Running PySpice simulator...")
    results = simulator_simulate(
        boxes=[hbm_box],
        bonding_box_list=[],
        TIM_boxes=[],
        heatsink_obj={"hc": 5000.0},
        layers=layers,
        power_dict={"HBM": 5.0}
    )
    
    # Check results
    print("\nResults:")
    if results:
        for name, (peak, avg, rx, ry, rz) in results.items():
            print("  {}:".format(name))
            print("    Peak Temperature: {:.2f}C".format(peak))
            print("    Avg Temperature:  {:.2f}C".format(avg))
            print("    Rx:               {:.6f} K/W".format(rx))
            print("    Ry:               {:.6f} K/W".format(ry))
            print("    Rz:               {:.6f} K/W".format(rz))
            
            # Physical plausibility checks
            try:
                assert peak > 25, "Peak temp should be > 25C (ambient), got {:.2f}C".format(peak)
                assert peak < 200, "Peak temp should be < 200C (unrealistic), got {:.2f}C".format(peak)
                assert avg <= peak, "Average should be <= peak, got avg={:.2f}C, peak={:.2f}C".format(avg, peak)
                print("  [PASS] Results are physically reasonable")
            except AssertionError as e:
                print("  [FAIL] Assertion failed: {}".format(e))
                return False
    else:
        print("   [FAIL] No results returned!")
        return False
    
    print("\n[PASS] TEST 2 PASSED\n")
    return True

def test_multiple_boxes():
    """Test with multiple boxes (GPU + HBM)"""
    print("=" * 60)
    print("TEST 3: Multiple Boxes (GPU 400W + HBM 5W)")
    print("=" * 60)
    
    # Create GPU box
    gpu_box = Box(
        start_x=0.0,
        start_y=0.0,
        start_z=0.0,
        width=30.0,
        length=30.0,
        height=1.0,
        power=400.0,
        stackup="1:Si",
        ambient_conduct=0.0,
        name="GPU"
    )
    
    # Create HBM box
    hbm_box = Box(
        start_x=30.0,
        start_y=0.0,
        start_z=0.0,
        width=10.0,
        length=10.0,
        height=1.0,
        power=5.0,
        stackup="1:Si",
        ambient_conduct=0.0,
        name="HBM"
    )
    
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
    
    # Run simulator
    print("Running PySpice simulator with multiple boxes...")
    results = simulator_simulate(
        boxes=[gpu_box, hbm_box],
        bonding_box_list=[],
        TIM_boxes=[],
        heatsink_obj={"hc": 5000.0},
        layers=layers,
        power_dict={"GPU": 400.0, "HBM": 5.0}
    )
    
    # Check results
    print("\nResults:")
    if results:
        for name, (peak, avg, rx, ry, rz) in results.items():
            print("  {}:".format(name))
            print("    Peak Temperature: {:.2f}C".format(peak))
            print("    Avg Temperature:  {:.2f}C".format(avg))
            print("    Rx:               {:.6f} K/W".format(rx))
            print("    Ry:               {:.6f} K/W".format(ry))
            print("    Rz:               {:.6f} K/W".format(rz))
            
            try:
                assert peak > 25, "Peak temp should be > 25C (ambient), got {:.2f}C".format(peak)
                assert peak < 200, "Peak temp should be < 200C (unrealistic), got {:.2f}C".format(peak)
                assert avg <= peak, "Average should be <= peak, got avg={:.2f}C, peak={:.2f}C".format(avg, peak)
                print("  [PASS] Results are physically reasonable")
            except AssertionError as e:
                print("  [FAIL] Assertion failed: {}".format(e))
                return False
    else:
        print("   [FAIL] No results returned!")
        return False
    
    print("\n[PASS] TEST 3 PASSED\n")
    return True

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING PYSPICE-BASED SIMULATOR INTEGRATION TESTS")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    try:
        all_passed = all_passed and test_single_box_gpu()
    except Exception as e:
        print("[FAIL] TEST 1 FAILED with exception: {}\n".format(e))
        all_passed = False
    
    try:
        all_passed = all_passed and test_single_box_hbm()
    except Exception as e:
        print("[FAIL] TEST 2 FAILED with exception: {}\n".format(e))
        all_passed = False
    
    try:
        all_passed = all_passed and test_multiple_boxes()
    except Exception as e:
        print("[FAIL] TEST 3 FAILED with exception: {}\n".format(e))
        all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("[PASS] ALL TESTS PASSED")
    else:
        print("[FAIL] SOME TESTS FAILED")
    print("=" * 60)
