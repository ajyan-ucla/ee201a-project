```python
#!/usr/bin/env python3
"""
End-to-end test: Run therm.py with minimal configuration
"""
import subprocess
import os
import sys

def run_simple_2p5d_config():
    """Run a simple 2.5D configuration"""
    print("=" * 60)
    print("E2E TEST: Simple 2.5D Configuration")
    print("=" * 60)
    
    # Using a simple config (you need to find which ones exist)
    cmd = [
        "python3", "therm.py",
        "--therm_conf", "configs/thermal-configs/sip_hbm_dray062325_1gpu_6hbm_2p5D.xml",
        "--out_dir", "out_test_e2e",
        "--heatsink_conf", "configs/thermal-configs/heatsink_definitions.xml",
        "--bonding_conf", "configs/thermal-configs/bonding_definitions.xml",
        "--heatsink", "heatsink_water_cooled",
        "--project_name", "test_e2e",
        "--is_repeat", "False",
        "--system_type", "2p5D_1GPU"
    ]
    
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        print(f"\nReturn code: {result.returncode}")
        
        if result.returncode == 0:
            # Check if output directory was created
            if os.path.exists("out_test_e2e"):
                print("✓ Output directory created")
                files = os.listdir("out_test_e2e")
                print(f"  Files: {files}")
            return True
        else:
            print("✗ Command failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Command timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"✗ Error running command: {e}")
        return False

if __name__ == "__main__":
    success = run_simple_2p5d_config()
    
    if success:
        print("\n" + "=" * 60)
        print("✓ E2E TEST PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ E2E TEST FAILED")
        print("=" * 60)
        sys.exit(1)
