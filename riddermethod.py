import math
import numpy as np
import matplotlib.pyplot as plt
import os

TOL_n = 10
TOL = 1e-10  # if wanting to change tolerance, change both TOL_n and TOL so that final rounding reflects this.
MAX_ITER = 20000  # Prevent overflow
ACTUAL_ROOT = -0.7034674225

def f(x):
    return math.exp(x) - (x ** 2)

def midpoint(a, b):
    return (a + b) / 2

def get_x3(x_0, x_1, x_2):
    return x_1 + (x_1 - x_0) * (np.sign(f(x_0)) * f(x_1)) / math.sqrt(f(x_1) ** 2 - f(x_0) * f(x_2))

# List of different intervals to test
intervals = [(-150, 150), (-100,100), (-20, 20), (-10, 10), (-5, 5), (-2, 2), (-1.5, 1.5), (-1.2, 1.2), (-1, 1), (-0.8, 0.8)]

plt.figure(figsize=(10, 6))
iteration_counts = []
interval_labels = []

for initial_x_0, initial_x_2 in intervals:
    print(f"Testing Interval: [{initial_x_0}, {initial_x_2}]")
    
    if (f(initial_x_0) * f(initial_x_2) <= 0):
        # there is a root in this interval
        x_0, x_2 = initial_x_0, initial_x_2
        x_3_prev = None
        errors = []
        
        for i in range(MAX_ITER):
            x_1 = midpoint(x_0, x_2)
            x_3 = get_x3(x_0, x_1, x_2)
            errors.append(abs(x_3 - ACTUAL_ROOT))
            
            if x_3_prev is not None and abs(x_3 - x_3_prev) < TOL:
                print(f"Converged to {round(x_3, TOL_n)} after {i + 1} iterations.")
                iteration_counts.append(i + 1)
                interval_labels.append(f"[{initial_x_0}, {initial_x_2}]")
                break
            else:
                x_3_prev = x_3
                if f(x_1) * f(x_3) < 0:
                    x_0, x_2 = x_1, x_3
                elif f(x_0) * f(x_3) < 0:
                    x_2 = x_3
                elif f(x_2) * f(x_3) < 0:
                    x_0 = x_3
        
        if i == MAX_ITER - 1:
            print("Stopped because MAX_ITER exceeded.")
        
        plt.plot(errors, marker='o', linestyle='-', label=f"Initial interval=[{initial_x_0}, {initial_x_2}]")
    else: print("Root does not exist in this interval")

plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title("Convergence Rate of Ridder's Method")
plt.legend()
plt.grid(True)
filename = "RiddersMethod.png"
plt.savefig(filename)
plt.close()
print(f"Plot saved at: {os.path.abspath(filename)}")

plt.figure(figsize=(13, 9))
plt.bar(interval_labels, iteration_counts)
plt.xlabel("Initial Interval")
plt.ylabel("Iterations to Converge")
plt.title("Ridder's Method Iterations to converge for each interval")
plt.yticks(range(0, max(iteration_counts) + 1))
plt.grid(axis='y', linestyle='-')
filename = "RiddersMethodIterations.png"
plt.savefig(filename)
plt.close()
print(f"Iteration count plot saved at: {os.path.abspath(filename)}")
