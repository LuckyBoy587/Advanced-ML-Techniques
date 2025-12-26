import subprocess
import os
import sys

def test_run():
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
    stdout, stderr = process.communicate(input="Sample.csv\n")
    
    # Check output
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    
    if "Confusion Matrix" in stdout:
        print("Test Passed: Confusion Matrix found.")
    else:
        print("Test Failed: Confusion Matrix not found.")

if __name__ == "__main__":
    test_run()
