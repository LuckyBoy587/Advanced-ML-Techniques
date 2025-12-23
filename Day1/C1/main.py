import ML_Modules
import os
import sys

def main():
    filename = input().strip()
    filename = os.path.join(sys.path[0], filename)
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return

    # Scenario 1
    # P_disease = 0.001
    # P_positive_given_disease = 0.95
    # P_positive_given_no_disease = 0.05
    # P_no_disease = 0.999
    print("Scenario 1: False positive rate = 5%")
    ML_Modules.calc_probability(0.001, 0.95, 0.05, 0.999)

    print() # Blank line separator

    # Scenario 2
    # P_disease = 0.001
    # P_positive_given_disease = 0.95
    # P_positive_given_no_disease = 0.10
    # P_no_disease = 0.999
    print("Scenario 2: False positive rate = 10%")
    ML_Modules.calc_probability(0.001, 0.95, 0.10, 0.999)

if __name__ == "__main__":
    main()
