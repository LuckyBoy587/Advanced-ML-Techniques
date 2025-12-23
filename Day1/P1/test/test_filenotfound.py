import subprocess
import os
import sys

def test_filenotfound():
    # Path to main.py
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'main.py')
    
    # Run the script with a non-existent file
    filename = "NonExistent.csv"
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(script_path)
    )
    
    stdout, stderr = process.communicate(input=f"{filename}\n")
    
    expected_output = f"Error: File '{filename}' not found.\n"
    
    print("STDOUT:")
    print(stdout)
    
    if expected_output.strip() in stdout.strip():
        print("Success: Error message verified.")
    else:
        print(f"Failure: Expected '{expected_output.strip()}', got '{stdout.strip()}'")

if __name__ == "__main__":
    test_filenotfound()
