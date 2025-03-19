# estimate the fixed point of T(x) to 10 decimal places, given initial x_0
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
import os

I_lower = 0.45
I_upper = 0.55
TOL = 1e-10
max_iter = 200
num_iter = 100
s = 0.472252 # actual fixed point (from wolfram alpha, found solution for T(x) = x)

def T(x):
    return 0.5*(math.cos(x/2) - abs(x-0.5))

def F(x):
    return x - 0.5*(math.cos(x/2) - abs(x-0.5))

def FPrime(x):
    return (4 + (2 *(-0.5 + x))/abs(-0.5 + x) + math.sin(x/2))/4

# tighten the interval provided so that fixed point iteration does not include where derivative DNE. Known that derivative DNE at 0.5
def bisection(lower, upper):
    n = 0
    for _ in range(max_iter):
        c = (lower + upper) / 2
        y = F(c)

        if ((lower > 0.5) or (upper < 0.5)):  # "bad" point is not in the interval
            break
        elif y > 0:
            upper = c  # Narrow the upper bound
            n += 1
        else:
            lower = c  # Narrow the lower bound
            n += 1

    return lower, upper, n

def FixedPoint_Estimate(x_0, iterations):
    list_xn = [0]*iterations
    list_estimates = [0]*iterations
    x_n = x_0
    for i in range (iterations):
        x_cur = T(x_n)
        list_xn[i] = x_n
        list_estimates[i] = x_cur
        x_n = x_cur
    return round(x_cur, 10), list_xn, list_estimates

def FixedPoint_EstimateWithPostEriori(x_0, TOL, max_iter): # initial guess, tolerance to 10 decimal places, and maximum allowed iterations (safeguard).
    x = x_0
    for i in range (max_iter):
        x_cur = T(x)
        if abs(x_cur - x) <TOL:
            # there is convergence
            return round(x_cur, 10), i+1
        x = x_cur
    print("Did not converge")

# returns an estimated required num of iterations given a starting approximation
def aPriori(x_0):
    x_1, listx, listy = FixedPoint_Estimate(x_0, 1)
    a = (1/(1-x_0))*abs(x_1 - x_0)
    return math.ceil((math.log(TOL/a))/(math.log(x_0))) 

def newtonsMethod(lower, upper, x_0):
    x_n = x_0
    list_n = [0] * max_iter
    list_x = [0] * max_iter
    for i in range(max_iter):
        x_npp = x_n - F(x_n) / FPrime(x_n)
        list_n[i] = i
        list_x[i] = x_npp
        if abs(x_npp - x_n) < TOL:
            print("\nFound convergence with Newton's Method:")
            plot_xy(list_n, list_x, "iteration", "x_n", "Newtons method iteration vs x_n")
            return round(x_npp, 10), i + 1, list_x[:i + 1]
        # did not find convergence yet, so keep iterating.
        
        x_n = x_npp

    # iterated out of bound of max iterations, don't want to overflow.
    print("Newton's Method did not converge")
    return round(x_npp, 10), i + 1, list_x[:i + 1]

def plot_xy(x_values, y_values, x_title, y_title, chart_title):
    plt.scatter(x_values, y_values, color='blue', alpha=0.6, edgecolors='black')

    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(chart_title)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.ticklabel_format(style='plain', axis='y', useOffset=False)
    plt.tight_layout()
    filename = "{}.png".format(chart_title.replace(" ", "_"))
    plt.savefig(filename)

    plt.close()

    print(f"Plot saved at: {os.path.abspath(filename)}")

def three_column_table(col1, col2, col3, col1name, col2name, col3name, title):
    data = {col1name: col1, col2name: col2, col3name: col3}
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 20))  # Adjust figure size as needed
    ax.xaxis.set_visible(False)  # Hide x-axis
    ax.yaxis.set_visible(False)  # Hide y-axis
    ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.box(on=None) # Remove the plot frame

    plt.savefig(title)
    plt.close()
    print(f"Plot saved at: {os.path.abspath(title)}")



# driver:

# need to modify the interval to avoid point where derivative DNE. Just perform bisection method until x=0.5 not included.

        # desmos graph here

a, b, n = bisection(0.45, 0.55)
print("Applying bisection method to [0.45, 0.55]:")
print("New interval to find a fixed point in: [{0}, {1}]".format(a, b))
print("(bisection method took", n, "iterations)\n")
# now have interval [a,b] which does not include problematic point where derivative = 0/DNE.
# show that the function is contractive on this interval to show that unique fixed point exists here.

        # paper work here


# Estimate fixed point, testing many starting approximations and testing performance of a priori (AP) vs post eriori (PE)

# get a set of x_0 in the interval where the fixed point exists. rounded to 10 decimal points.
List_x0 = [0] * num_iter
for i in range (num_iter): # not worrying about duplicates
    List_x0[i] = round(random.uniform(a,b), 10)

# lists for calculating difference from acecpted value for each estimate
list_differences_AP = [0] *num_iter
list_differences_PE = [0]* num_iter

# list for fixed point estimate for each x_0 using a priori
list_fixed_point_AP = [0]*num_iter

# list for estimating number of iterations required for each x_0
list_iteration_estimates = [0] * num_iter
for i in range(num_iter):
    list_iteration_estimates[i] = aPriori(List_x0[i])

colors = plt.cm.jet(np.linspace(0, 1, num_iter))  # distinct color for each set of x_0 and estimates.
plt.figure(figsize=(10, 6))

print("Performing fixed point iteration with a priori")
for i in range (num_iter):
    list_fixed_point_AP[i], list_xnAP, list_estimatesAP = FixedPoint_Estimate(List_x0[i], list_iteration_estimates[i]) # use a priori num of iterations
    list_differences_AP[i] = abs(s-list_fixed_point_AP[i])
    # Create a scatter plot showing convergence of each x_0 to same value:
    plt.scatter(list_xnAP, list_estimatesAP, color=colors[i], label=f"Set {i+1}")

plt.xlabel('x_0 and its Subsequent Iterations')
plt.ylabel('Estimate')
plt.title('Fixed Point Estimates\nEach color corresponds to a different x_0')
plt.grid(True)
filename = "APrioriLinearConvergence.png"
plt.savefig(filename)
plt.close()
print(f"Plot saved at: {os.path.abspath(filename)}")

#Graph x_0 vs a priori estimate of iterations needed.
plot_xy(List_x0, list_iteration_estimates, "x_0", "Predicted num of iterations to converge", "A Priori estimated iterations needed")

# Create table of x_0, its estimated number of iterations, and its estimate of fixed point.
three_column_table(List_x0, list_iteration_estimates, list_fixed_point_AP, "x_0", "A priori num of iterations", "Fixed pt Estimate", "APrioriTable.png")


# post eriori:
list_iterationsTaken_PE = [0]*num_iter
print("\nPerforming fixed point iteration with post eriori")
for i in range (num_iter):
    fixed_point_PE, list_iterationsTaken_PE[i] = FixedPoint_EstimateWithPostEriori(List_x0[i], TOL, max_iter)
    list_differences_PE[i] = abs(s-fixed_point_PE)
# for post eriori, graph x_0 and how many iterations it took to converge.
plot_xy(List_x0, list_iterationsTaken_PE, "x_0", "Iterations to Converge", "Post Eriori Starting Approximation vs Iterations to Converge")


# Determining the most accurate method: find the smallest difference between estimate and "accepted value" for each, smallest wins.
smallestDiff_AP = min(list_differences_AP)
smallestDiff_PE = min(list_differences_PE)
print("A priori estimate got within", smallestDiff_AP, "of accepted value")
print("Post eriori estimate got within", smallestDiff_PE, "of accepted value")
if (smallestDiff_AP > smallestDiff_PE):
    print("A priori estimate more accurate")
else:
    print("Post Eriori estimate more accurate")


print("\n4. Newton's method for T(x) in [0.45, 0.55]")
print("Difficulty: when f'(x) DNE, when dividing by f'(x) program will crash.")
print("Same solution as with fixed point iteration, perform bisection method until f'(x) DNE is not included, then need to check to make sure solution will still exist in new interval.")

NM_estimate, n, list_NM_estimates = newtonsMethod(a,b,a)
print("Newton's Method estimate:", NM_estimate, " in ", n, "iterations")