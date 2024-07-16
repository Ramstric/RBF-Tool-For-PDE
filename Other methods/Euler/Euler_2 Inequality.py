# Ramon Everardo Hernandez Hernandez | 19 october 2023 at 15:00
#               Euler's Method for solving differential equations

#                       Example 2 - Algorithm 5.1
# Given the equation y' = f(t, y) in the interval a <= t <= b and the initial condition y(a) = c
#

from math import e

# Inputs
a, b = [0, 2]  # Interval
N = 10  # Number of spaces
c = 0.5  # Initial condition

# Error Bound
L = 1
M = 0.5*pow(e, 2) - 2


def f(t: float, y: float):
    return y - pow(t, 2) + 1


def y_exact(t: float):
    return pow(t + 1, 2) - 0.5 * pow(e, t)


h = (b - a) / N  # Step size
t = a  # On the lower time limit -> t = a
w = c  # The initial condition -> y(t) = y(a) = c

# First iteration
actualError = [abs(y_exact(t) - w)]
errorBound = [((h*M)/(2*L))*(pow(e, (t-a)*L) - 1)]  # hM/2L * ( e^{(t - a)L} - 1 )

i = 1  # Starting on w_1

while i <= N:
    w = w + h * f(t, w)  # w_i+1 = w_i + h f(t_i, w_i)
    t = a + i * h  # t_i+1 = a + i h
    i += 1

    actualError.append(abs(y_exact(t) - w))
    errorBound.append(((h * M) / (2 * L)) * (pow(e, (t - a) * L) - 1))

actualError = [round(value, 5) for value in actualError]
errorBound = [round(value, 5) for value in errorBound]

for i in range(N+1):
    if not i:
        print(f"\n\t{'t_i':<15}|", end="")
    else:
        print(f"\t{i*h:<8.1f}", end="")

for i in range(N+1):
    if not i:
        print(f"\n\t{'':-<15}|", end="")
    else:
        print(f"\t{'-':-<8}", end="")

for i in range(N+1):
    if not i:
        print(f"\n\t{'Actual Error':<15}|", end="")
    else:
        print(f"\t{actualError[i]:<8.5f}", end="")

for i in range(N+1):
    if not i:
        print(f"\n\t{'Error Bound':<15}|", end="")
    else:
        print(f"\t{errorBound[i]:<8.5f}", end="")