# Ramon Everardo Hernandez Hernandez | 19 october 2023 at 15:48
#               Method of Runge Kutta (Order Four) for approximating
#               the solution of differential equations
#
#                       Example 3 | Table 5.8 | Algorithm 5.2
# Given the equation y' = f(t, y) in the interval a <= t <= b and the initial condition y(a) = c

from math import e

# Inputs
a, b = [0, 2]  # Interval
N = 10  # Number of spaces
c = 0.5  # Initial condition


def f(t: float, y: float):
    return y - pow(t, 2) + 1


def y_exact(t: float):
    return pow(t + 1, 2) - 0.5 * pow(e, t)


h = (b - a) / N  # Step size
t_i = a  # On the lower time limit -> t = a
w = c  # The initial condition -> y(t) = y(a) = c

i = 1  # Starting on w_1

print(f"\n\t{'':^15}|{'Exact':^15}|{'Runge-Kutta Ord 4':^19}|{'Error':^15}")
print(f"\t{'t_i':^15}|{'y_i':^15}|{'w_i':^19}|{'|y_i - w_i|':^15}")
print(f"\t{'':-^15}|{'':-^15}|{'':-^19}|{'':-^15}")
print(f"\t{t_i:^15.1f}|{y_exact(t_i):^15.7f}|{w:^19.7f}|{abs(y_exact(t_i) - w):^15.7f}")  # Values for w_0

while i <= N:

    k_1 = h * f(t_i, w)
    k_2 = h * f(t_i + (h / 2), w + (k_1 / 2))
    k_3 = h * f(t_i + (h / 2), w + (k_2 / 2))
    k_4 = h * f(t_i + h, w + k_3)

    w = w + (k_1 + 2*k_2 + 2*k_3 + k_4)/6
    t_i = a + i * h  # t_i+1 = a + i h
    i += 1

    print(f"\t{t_i:^15.1f}|{y_exact(t_i):^15.7f}|{w:^19.7f}|{abs(y_exact(t_i) - w):^15.7f}")
