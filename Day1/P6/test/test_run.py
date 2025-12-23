import subprocess
import os
import sys

def test_main():
    # Expected output matching the unwrapped format
    expected_output = """Creating dummy variables for salary:
   satisfaction_level  last_evaluation  number_project  average_montly_hours  time_spend_company  Work_accident  left  promotion_last_5years Department  salary  salary_low  salary_medium
0                0.38             0.53               2                   157                   3              0     1                      0      sales     low        True          False
1                0.80             0.86               5                   262                   6              0     1                      0      sales  medium       False           True
2                0.11             0.88               7                   272                   4              0     1                      0      sales  medium       False           True
3                0.72             0.87               5                   223                   5              0     1                      0      sales     low        True          False
4                0.37             0.52               2                   159                   3              0     1                      0      sales     low        True          False

[5 rows x 12 columns]

Creating dummy variables for department:
   satisfaction_level  last_evaluation  number_project  average_montly_hours  time_spend_company  Work_accident  left  promotion_last_5years Department  salary  salary_low  salary_medium  dept_accounting  dept_hr  dept_sales  dept_support  dept_technical
0                0.38             0.53               2                   157                   3              0     1                      0      sales     low        True          False            False    False        True         False           False
1                0.80             0.86               5                   262                   6              0     1                      0      sales  medium       False           True            False    False        True         False           False
2                0.11             0.88               7                   272                   4              0     1                      0      sales  medium       False           True            False    False        True         False           False
3                0.72             0.87               5                   223                   5              0     1                      0      sales     low        True          False            False    False        True         False           False
4                0.37             0.52               2                   159                   3              0     1                      0      sales     low        True          False            False    False        True         False           False

[5 rows x 17 columns]

Final dataframe with dummy variables:
   satisfaction_level  last_evaluation  number_project  average_montly_hours  time_spend_company  Work_accident  left  promotion_last_5years Department  salary  salary_low  salary_medium  dept_accounting  dept_hr  dept_sales  dept_support  dept_technical
0                0.38             0.53               2                   157                   3              0     1                      0      sales     low        True          False            False    False        True         False           False
1                0.80             0.86               5                   262                   6              0     1                      0      sales  medium       False           True            False    False        True         False           False
2                0.11             0.88               7                   272                   4              0     1                      0      sales  medium       False           True            False    False        True         False           False
3                0.72             0.87               5                   223                   5              0     1                      0      sales     low        True          False            False    False        True         False           False
4                0.37             0.52               2                   159                   3              0     1                      0      sales     low        True          False            False    False        True         False           False

[5 rows x 17 columns]

Size of training dataset: (36, 17)
Size of test dataset: (16, 17)

Shapes of input/output features after train-test split:
(36, 16) (36,) (16, 16) (16,)"""
    
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
        # Print first few chars of difference if any
        import difflib
        diff = difflib.ndiff(expected_output.splitlines(keepends=True), stdout.splitlines(keepends=True))
        print("Diff:")
        print(''.join(diff))

if __name__ == "__main__":
    test_main()
