import numpy as np
import sys, os
def entropy(y):
    if len(y) == 0:
        return 0.0
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum([p * np.log2(p) for p in probs if p > 0])
def information_gain(X_col, y, parent_entropy):
    left = y[X_col <= 0]
    right = y[X_col > 0]
    n = len(y)
    ig = parent_entropy
    for subset in (left, right):
        if len(subset) > 0:
            ig -= (len(subset) / n) * entropy(subset)
    return ig
filename = input().strip()
try:
    data = np.genfromtxt(os.path.join(sys.path[0], filename), delimiter=",", skip_header=1)
except:
    print(f"Error: Unable to read file '{filename}'.")
    sys.exit()
y = data[:, -1]
parent_entropy = entropy(y)
features = {
    "Fasting blood": data[:, 0],
    "bmi": data[:, 1],
    "FamilyHistory": data[:, 3]
}
print(f"Parent Node Entropy: {parent_entropy:.3f}")
ig_values = {}
for name, col in features.items():
    ig = information_gain(col, y, parent_entropy)
    ig_values[name] = ig
    print(f"Information Gain ({name}): {ig:.3f}")
best_feature = max(ig_values, key=ig_values.get)
print(f"Best Feature for root node: {best_feature} with Information Gain: {ig_values[best_feature]:.3f}")