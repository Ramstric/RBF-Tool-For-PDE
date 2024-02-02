# Ramon Everardo Hernandez Hernandez | 02 february 2024
#    Method of Finite-Difference for Linea Problems
#                     Revisited
#        Example 1 | Table 11.3 | Algorithm 11.3
#
#     Given the equation y'' = p(x) y' + q(x) y + r(x)
#                          in the interval a <= t <= b
# and the initial conditions y(a) = a_0 and y(b) = b_0

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

# mpl.use('Qt5Agg')

# Inputs
a, b = [1, 2]  # Interval
N = 9  # Number of spaces for discretization
alpha, beta = [1, 2]  # Boundary conditions


# ----[ Considering the general form of y'' = p(x) y' + q(x) y + r(x) ]----
def p(x):
    return -2/x


def q(x):
    return 2/pow(x, 2)


def r(x):
    return np.sin(np.log(x))/pow(x, 2)
# --------------------------------------------------------------------------


def main():
    h = (b - a) / (N + 1)  # Step size

    x = np.arange(a+h, b, h).round(1)  # Array of x points, from a to b, in h step size

    n = N-1

    # Pre allocating matrices
    A = np.zeros(shape=(N, N))
    B = np.zeros(shape=N)

    # A matrix building
    A[0, 0] = 2 + (pow(h, 2) * q(x[0]))
    A[0, 1] = - 1 + ((h/2) * p(x[0]))

    A[n, n] = 2 + (pow(h, 2) * q(x[n]))
    A[n, n-1] = - 1 - ((h/2) * p(x[n]))

    # B matrix building
    B[0] = - (pow(h, 2) * r(x[0])) + (1 + (h / 2) * p(x[0])) * alpha
    B[n] = - (pow(h, 2) * r(x[n])) + (1 - ((h / 2) * p(x[n]))) * beta

    for i in range(1, N-1):
        A[i, i-1] = - 1 - ((h/2) * p(x[i]))
        A[i, i] = 2 + (pow(h, 2) * q(x[i]))
        A[i, i+1] = - 1 + ((h/2) * p(x[i]))

        B[i] = - pow(h, 2) * r(x[i])

    W = np.linalg.solve(A, B)  # Approximated numerical solutions

    # Exact numerical solutions
    y = np.zeros(shape=N)
    for i in range(1, N - 1):
        y[i] = i

    Table_1 = pd.DataFrame(data=[x.round(2), W, y], index=['x', 'w', 'y']).T

    print(W)


main()