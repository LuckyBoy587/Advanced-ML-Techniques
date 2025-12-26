import subprocess
import os
import sys

def test_filenotfound():
    # Run the main.py script with non-existent file
    process = subprocess.Popen(
        [sys.executable, 'main.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    
    stdout, stderr = process.communicate(input="NonExistent.csv\n")
    
    # Normalize line endings
    stdout = stdout.strip()
    expected = "Error: File 'NonExistent.csv' not found."
    
    print("STDOUT:", stdout)
    if stderr:
        print("STDERR:", stderr)
    
    if stdout == expected:
        print("Test PASSED")
    else:
        print("Test FAILED")
        print(f"Expected: '{expected}'")
        print(f"Got: '{stdout}'")

if __name__ == "__main__":
    test_filenotfound()

