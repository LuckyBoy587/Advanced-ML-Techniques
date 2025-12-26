import subprocess
import os
import sys

def test_filenotfound():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to get to the main script directory
    script_dir = os.path.join(current_dir, '..')
    main_script = os.path.join(script_dir, 'main.py')
    
    # Run the main script
    process = subprocess.Popen(
        [sys.executable, main_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=script_dir, # Run in the directory where main.py is
        text=True
    )
    
    # Send input
    stdout, stderr = process.communicate(input="NonExistentFile.csv\n")
    
    # Check output
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    
    if "Error: File" in stdout and "not found" in stdout:
        print("Test Passed: File Not Found error handled.")
    else:
        print("Test Failed: File Not Found error not handled correctly.")

if __name__ == "__main__":
    test_filenotfound()
