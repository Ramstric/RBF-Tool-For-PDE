# Ramon Everardo Hernandez Hernandez | 08 december 2023 at 13:00
#               Method of Finite-Difference for Linea Problems
#
#                 Example 1 | Table 11.3 | Algorithm 11.3
#
#   Given the equation y'' = p(x) y' + q(x) y + r(x)
#         in the interval a <= t <= b
#         and the initial conditions y(a) = a_0 and y(b) = b_0

import math

# Inputs
a, b = [1, 2]  # Interval
N = 9  # Number of spaces
alpha, beta = [1, 2]  # Boundary conditions


def p(x: float):
    return -2/x


def q(x: float):
    return 2/pow(x, 2)


def r(x: float):
    return math.sin(math.log(x))/pow(x, 2)


# Step 1
h = (b - a) / (N + 1)  # Step size
x_i = a + h

a_mat = [2 + pow(h, 2) * q(x_i)]
b_mat = [-1 + (h/2) * p(x_i)]
c_mat = [0]
d_mat = [-pow(h, 2) * r(x_i) + (1 + (h/2) * p(x_i)) * alpha]

# Step 2. For i = 2 up to N-1...
for i in range(2, N):
    x_i = a + i*h
    a_mat.append(2 + pow(h, 2) * q(x_i))
    b_mat.append(-1 + (h/2) * p(x_i))
    c_mat.append(-1 - (h/2) * p(x_i))
    d_mat.append(-pow(h, 2) * r(x_i))

# Step 3
x_i = b - h
a_mat.append(2 + pow(h, 2) * q(x_i))
b_mat.append(0)
c_mat.append(-1 - (h/2) * p(x_i))
d_mat.append(-pow(h, 2) * r(x_i) + (1 - (h/2) * p(x_i)) * beta)


# Step 4. Solution of a tridiagonal linear system
l_mat = [a_mat[0]]
u_mat = [b_mat[0]/a_mat[0]]
z_mat = [d_mat[0]/l_mat[0]]

# Step 5. For i = 2 up to N-1...
for i in range(1, N-1):
    l_mat.append(a_mat[i] - c_mat[i]*u_mat[i-1])
    u_mat.append(b_mat[i]/l_mat[i])
    z_mat.append((d_mat[i] - c_mat[i]*z_mat[i-1])/l_mat[i])

# Step 6
l_mat.append(a_mat[N-1] - c_mat[N-1]*u_mat[N-2])
u_mat.append(0)
z_mat.append((d_mat[N-1] - c_mat[N-1]*z_mat[N-2])/l_mat[N-1])

# Step 7
w_mat = list(range(N+2))

w_mat[0] = alpha
w_mat[N+1] = beta
w_mat[N] = z_mat[N-1]

# Step 8
for i in reversed(range(N-1)):
    w_mat[i+1] = z_mat[i] - u_mat[i]*w_mat[i+2]

print(f"\t{'x_i':^15}|{'w_i':^15}|{'y_i':^19}|{'|y_i - w_i|':^15}")
print(f"\t{'':-^15}|{'':-^15}|{'':-^19}|{'':-^15}")

# Step 9
for i in range(N+1):
    x_i = a + i*h
    print(f"\t{x_i:^15.1f}|{w_mat[i]:^15.8f}|{w_mat[i]:^19.7f}|{abs(w_mat[i] - 0):^15.7f}")  # Values for w_0
