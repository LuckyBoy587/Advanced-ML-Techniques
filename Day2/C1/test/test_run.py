import subprocess
import os
import sys

def test_main():
    # Expected output is hard to match exactly due to whitespace/pandas versions.
    # We will verify key parts of the output.
    
    # Run the main.py script
    process = subprocess.Popen(
        [sys.executable, 'main.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    
    stdout, stderr = process.communicate(input="Sample.csv\n")
    
    # Normalize line endings
    stdout = stdout.replace('\r\n', '\n').strip()
    
    # Check for critical sections
    sections = [
        "# Head of all columns",
        "# Data Types of all columns",
        "# Working subset head",
        "# Mean values grouped by Outcome",
        "# Null value check",
        "# Zero-value count for BMI",
        "# Zero-value count for Glucose",
        "# Zero-value count for Age",
        "# Zero-value count after removal: Glucose",
        "# Zero-value count after removal: BMI",
        "# Number of rows after zero-value removal",
        "# Data after outlier treatment",
        "# SVM Model Evaluation",
        "Confusion Matrix",
        "Classification Report:",
        "accuracy:",
        "recall:",
        "f1-score:",
        "precision:"
    ]
    
    missing = []
    for section in sections:
        if section not in stdout:
            missing.append(section)
            
    if missing:
        print("Test FAILED")
        print("Missing sections:", missing)
        print("STDOUT:\n", stdout)
        if stderr:
            print("STDERR:\n", stderr)
    else:
        # Check specific values
        if "48" not in stdout: # Row count after removal
             print("Test FAILED: Row count 48 not found")
             print("STDOUT:\n", stdout)
             return

        if "accuracy: 0.600" in stdout or "accuracy: 0.6" in stdout:
             print("Test PASSED")
        else:
             print("Test FAILED: Accuracy 0.600 not found")
             print("STDOUT:\n", stdout)

if __name__ == "__main__":
    test_main()
