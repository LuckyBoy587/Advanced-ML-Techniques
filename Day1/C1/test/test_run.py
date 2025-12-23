import subprocess
import os

def test_main():
    # Expected output
    expected_output = """Scenario 1: False positive rate = 5%
Probability of having disease given a positive test: 0.0187

Scenario 2: False positive rate = 10%
Probability of having disease given a positive test: 0.0094
"""
    
    # Run the main.py script
    process = subprocess.Popen(
        ['python', 'main.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Go up one level to project root
    )
    
    stdout, stderr = process.communicate(input="Sample.csv\n")
    
    # Normalize line endings
    stdout = stdout.replace('\r\n', '\n').strip()
    expected_output = expected_output.strip()
    
    print("STDOUT:", stdout)
    if stderr:
        print("STDERR:", stderr)
        
    if stdout == expected_output:
        print("Test PASSED")
    else:
        print("Test FAILED")
        print("Expected:\n", expected_output)
        print("Got:\n", stdout)

if __name__ == "__main__":
    test_main()
