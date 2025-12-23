import subprocess
import os
import sys

def test_main():
    # Expected output matching the truncated format seen in the test environment
    expected_output = """Rows with missing values (if any):
   satisfaction_level  last_evaluation  ...  Department  salary
2                0.11              7.0  ...         NaN     NaN

[1 rows x 10 columns]

Correlation matrix of numeric columns:
                      satisfaction_level  ...      left
satisfaction_level              1.000000  ...  0.191038
last_evaluation                -0.133676  ... -0.979890
number_project                 -0.197009  ... -0.998953
average_montly_hours            0.087538  ...  0.396414
time_spend_company              0.585999  ...  0.454941
Work_accident                  -0.135261  ... -0.700140
left                            0.191038  ...  1.000000

[7 rows x 7 columns]"""
    
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'main.py')
    
    # Run the main.py script
    process = subprocess.Popen(
        [sys.executable, script_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.path.dirname(script_path)
    )
    
    stdout, stderr = process.communicate(input="Sample.csv\n")
    
    # Normalize line endings and strip
    stdout = stdout.replace('\r\n', '\n').strip()
    expected_output = expected_output.replace('\r\n', '\n').strip()
    
    if stdout == expected_output:
        print("Test PASSED")
    else:
        print("Test FAILED")
        print("Expected:\n", expected_output)
        print("Got:\n", stdout)

if __name__ == "__main__":
    test_main()