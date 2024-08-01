# Ramon Everardo Hernandez Hernandez | 23 october 2023 at 16:15
#               Method of Poisson Equation Finite-Difference
#                   for Non-Linear Problems
#
#                 Example 2 | Table 12.2 | Algorithm 12.1
#
#   Given the equation
#         in the interval a <= t <= b
#         and the initial conditions y(a) = a_0 and y(b) = b_0

import math
from colorama import Fore, Back

# Inputs
a, b, c, d = [0, math.pi, 0, math.pi/2]  # Endpoints
m, n = [5, 5]  # Integers where m >= 3 and n >= 3
TOL = 0.0000000001  # Tolerance
N = 30  # Maximum number of iterations


def f(x: float, y: float):
    return -(math.cos(x+y)+math.cos(x-y))


def g(x: float, y: float):
    if x == a:
        return math.cos(y)
    elif x == b:
        return -math.cos(y)
    elif y == c:
        return math.cos(x)
    elif y == d:
        return 0


# Step 1
h = (b - a) / n  # Step size for one axis
k = (d - c) / m  # Step size for the other axis

# Steps 2 & 3. Mesh points
x = [0] * (n - 1)

for i in range(n - 1):
    x[i] = a + (i+1)*h

y = [0] * (m - 1)

for j in range(m - 1):
    y[j] = c + (j+1)*k

w = list()
# Step 4
for i in range(n - 1):
    w.append(list())
    for j in range(m - 1):
        w[i].append(0)

# Step 5
lamb = pow(h, 2)/pow(k, 2)
mu = 2*(1 + lamb)
l = 1

# Step 6
while l <= N:

    # Step 7
    z = (-pow(h, 2) * f(x[0], y[m - 2]) + g(a, y[m - 2]) + lamb * g(x[0], d) + lamb * w[0][m - 3] + w[1][m - 2])/mu
    NORM = abs(z - w[0][m - 2])

    w[0][m - 2] = z

    for i in range(1, n - 2):
        z = (-pow(h, 2) * f(x[i], y[m - 2]) + lamb * g(x[i], d) + w[i - 1][m - 2] + w[i + 1][m - 2] + lamb * w[i][m - 3])/mu
        print(z)
        if abs(w[i][m - 2] - z) > NORM:
            NORM = abs(w[i][m - 2] - z)

        w[i][m - 2] = z

        # Step 9
        z = (-pow(h, 2) * f(x[n - 2], y[m - 2]) + g(b, y[m - 2]) + lamb * g(x[n - 2], d) + w[n - 3][m - 2] + lamb * w[n - 2][m - 3]) / mu

        if abs(w[n - 2][m - 2] - z) > NORM:
            NORM = abs(w[n - 2][m - 2] - z)

        w[n - 2][m - 2] = z  # <----- LAST EDIT

        # Step 10
        for j in reversed(range(1, m - 2)):
            # Step 11
            z = (-pow(h, 2) * f(x[0], y[j]) + g(a, y[j]) + lamb * w[0][j + 1] + lamb * w[0][j - 1] + w[1][j] ) / mu

            if abs(w[0][j] - z) > NORM:
                NORM = abs(w[0][j] - z)

            w[0][j] = z

            # Step 12
            for i_2 in range(1, n - 2):
                z = (-pow(h, 2) * f(x[i_2], y[j]) + w[i_2 - 1][j] + lamb * w[i_2][j + 1] + w[i_2 + 1][j] + lamb * w[i_2][j - 1]) / mu

                if abs(w[i_2][j] - z) > NORM:
                    NORM = abs(w[i_2][j] - z)

                w[i_2][j] = z

            # Step 13
            z = (-pow(h, 2) * f(x[n - 2], y[j]) + g(b, y[j]) + w[n - 3][j] + lamb * w[n - 2][j + 1] + lamb * w[n - 2][j - 1]) / mu

            if abs(w[n - 2][j] - z) > NORM:
                NORM = abs(w[n - 2][j] - z)

            w[n - 2][j] = z

        # Step 14
        z = (-pow(h, 2) * f(x[0], y[0]) + g(a, y[0]) + lamb * g(x[0], c) + lamb * w[0][1] + w[1][0]) / mu

        if abs(w[0][0] - z) > NORM:
            NORM = abs(w[0][0] - z)

        w[0][0] = z

        # Step 15
        for i_2 in range(1, n - 2):
            z = (-pow(h, 2) * f(x[i_2], y[0]) + lamb * g(x[i_2], c) + w[i_2 - 1][0] + lamb * w[i_2][1] + w[i_2 + 1][0]) / mu

            if abs(w[i_2][0] - z) > NORM:
                NORM = abs(w[i_2][0] - z)

            w[i_2][0] = z

        # Step 16
        z = (-pow(h, 2) * f(x[n - 2], y[0]) + g(b, y[0]) + lamb * g(x[n - 2], c) + w[n - 3][0] + lamb * w[n - 2][1]) / mu

        if abs(w[n - 2][0] - z) > NORM:
            NORM = abs(w[n - 2][0] - z)

        w[n - 2][0] = z

        # Step 17
        if NORM <= TOL:

            # Step 18
            for i_2 in range(n - 1):
                for j_2 in range(m - 1):
                    print(f"x[{i_2}] = {x[i_2]} \ty[{j_2}] = {y[j_2]:0.1f} \tw[{i_2}][{j_2}] = {w[i_2][j_2]}")

            # Step 19
            break

    # Step 20
    l = l + 1

else:
    # Step 21
    print("\n\nMaximum number of iterations exceeded\n")


print(Back.BLACK + f"\t{'i':^3}{'j':^3}|{'x_i':^15}|{'y_j':^15}|{'w_i,j':^15}" + Back.RESET)
print(f"\t{'':-^3}{'':-^3}|{'':-^15}|{'':-^15}|{'':-^15}")

for i in range(n - 1):
    for j in range(m - 1):
        if i == 1 and j == 0:
            print(Fore.RED, end="")
        elif i == 1 and j == 2:
            print(Fore.BLUE, end="")
        elif i == 3 and j == 0:
            print(Fore.MAGENTA, end="")
        elif i == 3 and j == 2:
            print(Fore.GREEN, end="")
        print(f"\t{i+1:^3}{j+1:^3}|{x[i]:^15.7f}|{y[j]:^15.7f}|{w[i][j]:^15.7f}")
        print(Fore.RESET, end="")


print("\n\nProcedure was successful!")
