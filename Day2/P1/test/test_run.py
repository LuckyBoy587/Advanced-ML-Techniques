import subprocess
import os
import sys

def test_run_main():
    # Get the directory of the current script (test_run.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # The main.py is in the parent directory of 'test' directory
    script_dir = os.path.dirname(current_dir)
    script_path = os.path.join(script_dir, 'main.py')
    
    # Run the script
    # We need to set cwd to script_dir so Sample.csv is found relative to it if needed
    # But the script uses sys.path[0] which should resolve correctly regardless of cwd if run as file
    # Input is 'Sample.csv'
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=script_dir # Set CWD to where Sample.csv is likely located
    )
    
    stdout, stderr = process.communicate(input="Sample.csv\n")
    
    if process.returncode != 0:
        print("Script failed with error:")
        print(stderr)
        sys.exit(1)
        
    print("STDOUT:")
    print(stdout)
    
    # Basic assertions
    assert "=== First 5 Rows of Data ===" in stdout
    assert "The number of samples in data is 50." in stdout
    assert "=== Data Types ===" in stdout
    assert "satisfaction_level       float64" in stdout
    assert "salary                    object" in stdout
    assert "=== Statistical Summary (Describe) ===" in stdout
    assert "=== Missing Values Per Column ===" in stdout
    assert "=== Salary Encoding Classes ===" in stdout
    assert "['low', 'medium']" in stdout
    assert "=== Department Encoding Classes ===" in stdout
    assert "['accounting', 'hr', 'sales', 'support', 'technical']" in stdout
    assert "=== Dropping 'Department' and 'salary' columns ===" in stdout
    assert "=== Updated DataFrame Info ===" in stdout
    assert "salary.enc" in stdout
    assert "Department.enc" in stdout

if __name__ == "__main__":
    test_run_main()
