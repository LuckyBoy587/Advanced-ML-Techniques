import subprocess
import os
import sys

def test_filenotfound():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.dirname(current_dir)
    script_path = os.path.join(script_dir, 'main.py')
    
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=script_dir
    )
    
    filename = "NonExistentFile.csv"
    stdout, stderr = process.communicate(input=f"{filename}\n")
    
    print("STDOUT:")
    print(stdout)
    
    assert f"Error: File '{filename}' not found." in stdout

if __name__ == "__main__":
    test_filenotfound()

