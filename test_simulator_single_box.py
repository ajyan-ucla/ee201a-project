#!/usr/bin/env python3
"""
Integration test: Single box thermal simulation
Simplest possible test case
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator import simulator_simulate
from rearrange import Box
from therm_xml_parser import Layer

def test_single_box_gpu():
    """Test with single GPU box"""
    print("=" * 50)
    print("TEST 1: Single GPU Box (400W)")
    print("=" * 50)
    
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
    
    # Create mock heatsink
    heatsink_obj = {"hc": 5000.0}  # 5000 W/(m²·K)
    
    # Run simulator
    print("Running simulator...")
    results = simulator_simulate(
        boxes=[gpu_box],
        bonding_box_list=[],
        TIM_boxes=[],
        heatsink_obj=heatsink_obj,
        layers=layers
    )
    
    # Check results
    print("\nResults:")
    if results:
        for name, (peak, avg, Rx, Ry, Rz) in results.items():
            print(f"  {name}:")
            print(f"    Peak Temp: {peak:.2f}C")
            print(f"    Avg Temp:  {avg:.2f}C")
            print(f"    Rx:        {Rx:.6f} K/W")
            print(f"    Ry:        {Ry:.6f} K/W")
            print(f"    Rz:        {Rz:.6f} K/W")
            
            # Sanity checks
            assert peak > 25, f"Peak temp should be > 25C (ambient), got {peak}"
            assert peak < 200, f"Peak temp should be < 200C (unrealistic), got {peak}"
            assert avg <= peak, f"Average should be <= peak, got avg={avg}, peak={peak}"
            
            print(f"   Results are physically reasonable")
    else:
        print("   No results returned!")
        return False
    
    print("\n TEST PASSED")
    return True

def test_single_box_hbm():
    """Test with single HBM box"""
    print("\n" + "=" * 50)
    print("TEST 2: Single HBM Box (5W)")
    print("=" * 50)
    
    # Create a single HBM box
    # ~10mm x 10mm x 1mm, 5W power
    hbm_box = Box(
        start_x=0.0,
        start_y=0.0,
        start_z=0.0,
        width=10.0,
        length=10.0,
        height=1.0,
        power=5.0,  # 5W (much lower than GPU)
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
    
    heatsink_obj = {"hc": 5000.0}
    
    print("Running simulator...")
    results = simulator_simulate(
        boxes=[hbm_box],
        bonding_box_list=[],
        TIM_boxes=[],
        heatsink_obj=heatsink_obj,
        layers=layers
    )
    
    print("\nResults:")
    if results:
        for name, (peak, avg, Rx, Ry, Rz) in results.items():
            print(f"  {name}:")
            print(f"    Peak Temp: {peak:.2f}C")
            print(f"    Avg Temp:  {avg:.2f}C")
            
            # With 5W, should be much closer to ambient
            assert peak < 100, f"HBM with 5W should be < 100C, got {peak}"
            assert peak > 25, f"Should still be above ambient"
            
            print(f"   Results are reasonable for low-power HBM")
    else:
        print("   No results!")
        return False
    
    print("\n TEST PASSED")
    return True

if __name__ == "__main__":
    try:
        test1 = test_single_box_gpu()
        test2 = test_single_box_hbm()
        
        if test1 and test2:
            print("\n" + "=" * 50)
            print(" ALL INTEGRATION TESTS PASSED")
            print("=" * 50)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
