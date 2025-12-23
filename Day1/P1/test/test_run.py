import subprocess
import os
import sys

def test_main():
    # Path to main.py
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'main.py')
    
    # Run the script
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(script_path) # Run from the directory where main.py is
    )
    
    stdout, stderr = process.communicate(input="Sample.csv\n")
    
    print("STDOUT:")
    print(stdout)
    print("STDERR:")
    print(stderr)

    if process.returncode != 0:
        print("Process failed.")

if __name__ == "__main__":
    test_main()

