# Ramon Everardo Hernandez Hernandez | 02 february 2024
#    Method of Finite-Difference for Linea Problems
#                     Revisited
#        Example with exact solution, and plotting

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

# mpl.use('Qt5Agg')

# Inputs
a, b = [-0.2, 0.05]  # Interval
N = 24  # Number of spaces for discretization
alpha, beta = [1.6353, 1.7043]  # Boundary conditions


# ----[ Considering the general form of y'' = p(x) y' + q(x) y + r(x) ]----
def p(x):
    return 4


def q(x):
    return 12


def r(x):
    return 3*np.exp(5*x)
# --------------------------------------------------------------------------


# Exact solution function
def y(x):
    return np.exp(6*x) + np.exp(-2*x) - (3/7)*np.exp(5*x)


def main():
    h = (b - a) / (N + 1)  # Step size

    x = np.arange(a+h, b, h).round(4)  # Array of x points, from a to b, in h step size

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
    Y = np.zeros(shape=N)
    for i in range(N):
        Y[i] = y(x[i])

    Table_1 = pd.DataFrame(data=[x, W, Y], index=['x', 'w', 'y']).T
    print(Table_1)

    # Plotting exact solution
    x_plot = np.linspace(a, b, num=50)
    y_plot = np.zeros(np.size(x_plot))

    for i in range(x_plot.size):
        y_plot[i] = y(x_plot[i])

    plt.scatter(x, W)                     # Approximate solution scatter
    plt.plot(x_plot, y_plot)        # Exact solution plot
    plt.show()


main()
