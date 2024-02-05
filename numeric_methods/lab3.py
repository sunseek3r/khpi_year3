import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize, linewidth=600, precision = 2, suppress=True)
n = 20
m = 10


def f(x, y):
    return np.abs(x) + (y**2)/2


def init():
    """
    Ініціалізує початкову матрицю
    :return:
    matrix - початкова матриця
    """
    matrix = np.zeros(shape=(m + 1, n + 1))
    x0 = np.linspace(0, 4, n + 1)
    y0 = np.linspace(0, 2, m + 1)
    # Fill the grid with initial conditions
    for i in range(0, n + 1):
        matrix[int(np.sqrt(4 - x0[i]) * n/4), i] = f(x0[i], np.sqrt(4-x0[i]))
    for j in range(0, m + 1):
        matrix[j, int(m * (1 - y0[j] ** 2 / 4))] = f(4-y0[j]**2, y0[j])
    return matrix


def grid_method(u):
    new_u = u.copy()
    while True:
        for i in range(1, n):
            for j in range(1, m):
                if u[j + 1, i] != 0 and u[j, i + 1] != 0:
                    new_u[j, i] = 1 / 4 * (u[j + 1, i] + u[j - 1, i] + u[j, i + 1] + u[j, i - 1])
        for i in range(1, n):
            new_u[0, i] = 1 / 4 * (2 * u[1, i] + u[0, i + 1] + u[0, i - 1])
        for j in range(1, m):
            new_u[j, 0] = 1 / 4 * (u[j + 1, 0] + u[j - 1, 0] + 2 * u[j, 1])
        new_u[0, 0] = 1 / 2 * (u[0, 1] + u[1, 0])
        if np.allclose(u, new_u, atol=0.0001):
            break
        u = new_u.copy()

    return u


def print_grid(u):
    # Round and print the values at grid nodes
    print(u)


def plot_3d_grid(u):
    Z = u
    x0 = np.linspace(0, 4, n + 1)
    y0 = np.linspace(0, 2, m + 1)
    X, Y = np.meshgrid(x0, y0)

    X = np.vstack((np.flipud(X), X))
    Y = np.vstack((np.flipud(Y), -Y))
    X = np.hstack((np.fliplr(X), -X))
    Y = np.hstack((np.fliplr(Y), Y))
    Z = np.vstack((np.flipud(Z), Z))
    Z = np.hstack((np.fliplr(Z), Z))

    print(Z)
    with open("matrix.txt", "w+") as fil:
        fil.write(str(Z))
    non_zero_mask = Z == 0
    X = np.ma.masked_where(non_zero_mask, X)
    Y = np.ma.masked_where(non_zero_mask, Y)
    Z = np.ma.masked_where(non_zero_mask, Z)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('u')
    plt.show()


u = init()
u = grid_method(u)
# print_grid(u)
plot_3d_grid(u)

