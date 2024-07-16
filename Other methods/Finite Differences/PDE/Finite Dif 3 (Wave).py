# Ramon Everardo Hernandez Hernandez | 13 december 2023
#               Method of Crank-Nicolson
#               for Non-Linear Problems
#
#           Example 3 | Table 12.5 | Algorithm 12.3
#
#   Given the equation
#         in the interval 0 < x < l and 0 < t < T
#
#         subject to the boundary conditions
#               u(0, t) = u(l, t) = 0 for 0 < t < T
#
#         and the initial conditions
#               u(x, 0) = f(x) for 0 < x < l

import math

# Inputs
m, N = [10, 50]   # Integers where m >= 3 and n >= 3
alph = 1        # Constant of parabolic PD equation

l = 1           # Endpoint (Limit of x)
T = 5           # Maximum time (Limit of t)


def f(x: float):
    return math.sin(math.pi * x)


# Step 1
h = l / m  # Step size for one axis
k = T / N  # Step size for the other axis
lamb = (pow(alph, 2) * k)/pow(h, 2)

w = l = u = [0] * (m - 1)
w[m] = 0

# Step 2
for i in range(m - 1):
    w[i] = f(i*h)

# Steps 3. Solving the tridiagonal linear system
l[0] = 1 + lamb
u[0] = - lamb/(2 * l[0])

# Step 4
for i in range(m - 2):
    l[i] = 1 + lamb + lamb * (u[i-1]/2)
    u[i] = - lamb/(2 * l[i])

# Step 5
l[m-1] = 1 + lamb + lamb * (u[m-2]/2)

z = [0] * (m - 1)

# Step 6
for i in range(N):

    # Step 7
    t = i*k
    z[0] = ((1 - lamb) * w[0] + (lamb/2) * w[1])/l[0]

    # Step 8
    for j in range(m - 1):
        z[j] = ((1 - lamb) * w[j] + (lamb/2) * (w[j+1] + w[j-1] + z[j-1]))/l[j]

    # Step 9
    w[m-1] = z[m-1]

    # Step 10
    for j in range(m - 2):
        w[j] = z[j] - u[j]*w[j+1]

    # Step 11
    print(t)

    for j in range(m - 1):
        x = j*h
        print(x)
        print(w[j])


print(f"\t{'i':^3}{'j':^3}|{'x_i':^15}|{'y_j':^15}|{'w_i,j':^15}")
print(f"\t{'':-^3}{'':-^3}|{'':-^15}|{'':-^15}|{'':-^15}")
print(f"\t{i+1:^3}{j+1:^3}|{x[i]:^15.7f}|{y[j]:^15.7f}|{w[i][j]:^15.7f}")
