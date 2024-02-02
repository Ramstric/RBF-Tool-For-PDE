# Ramon Everardo Hernandez Hernandez | 18 de octubre de 2023 a las 15:26
#           Metodo de Euler para resolver ecuaciones diferenciales
#                       Ejemplo 1 - Algoritmo 5.1
# Para la ecuación y' = f(t, y) en el intervalo a <= t <= b y con CI y(a) = c
#

from math import e

# Entradas
a, b = [0, 2]  # Intervalo
N = 10  # Cantidad de espacios
c = 0.5  # Condicion inicial


def f(t: float, y: float):
    return y - pow(t, 2) + 1


def yExacta(t: float):
    return pow(t + 1, 2) - 0.5 * pow(e, t)


h = (b - a) / N  # Definicion del step size
t = a  # En el tiempo inferior a
w = c  # Tenemos la condicion inicial y(t) = y(a) = c

print(f"\n\t{'t_i':^15}|{'w_i':^15}|{'y_i':^15}|{'|y_i - w_i|':^15}")
print(f"\t{'':-^15}|{'':-^15}|{'':-^15}|{'':-^15}")
print(f"\t{t:^15}|{w:^15.7f}|{yExacta(t):^15.7f}|{abs(yExacta(t) - w):^15.7f}")  # Valores para w_0

i = 1  # Empieza en w_1

while i <= N:
    w = w + h * f(t, w)  # w_i+1 = w_i + h f(t_i, w_i)
    t = a + i * h  # t_i+1 = a + i h
    i += 1

    t = round(t, 7)
    w = round(w, 7)
    print(f"\t{t:^15}|{w:^15.7f}|{yExacta(t):^15.7f}|{abs(yExacta(t) - w):^15.7f}")
