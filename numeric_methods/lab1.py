# Визначення функції f(x), яка буде використовуватися для початкових умов

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 6


def f(x):
    return np.sin(x + 0.45)


def f_t(t):
    return 0.435 - 2*t


def init():
    # Initialize variables
    matrix = np.zeros(shape=(n+1, n+1))
    row0 = np.linspace(0, 0.6, n+1)
    t = np.linspace(0, 0.01, n+1)
    # Fill the grid with initial conditions
    matrix[0, :] = f(row0)
    matrix[:, 0] = f_t(t)
    matrix[:, n] = 0.8674
    return matrix


def grid_method(u):
    # Calculate values at grid nodes
    for j in range(0, n):
        for i in range(1, n):
            u[j + 1, i] = 1 / 6 * (u[j, i + 1] + 4 * u[j, i] + u[j, i - 1])
    return u


def print_grid(u):
    # Round and print the values at grid nodes
    np.round(u, 4)
    print(u)


def plot_3d_grid(u):
    row0 = np.linspace(0, 0.6, n+1)
    t = np.linspace(0, 0.01, n+1)
    x, y = np.meshgrid(row0, t)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, u, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    plt.show()


u = init()
u = grid_method(u)
print_grid(u)
plot_3d_grid(u)
