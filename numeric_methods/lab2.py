import numpy as np
import matplotlib.pyplot as plt

n = 10
m = 5
h = 0.1


def f(x):
    return 0.5*(x**2 + 1)


def F(x):
    return x * np.sin(2*x)


def fi(t):
    return 0.5 + 3 * t


def psi(t):
    return 1


def init():
    """
    Ініціалізує початкову матрицю
    :return:
    matrix - початкова матриця
    """
    matrix = np.zeros(shape=(n+1, m+1))
    r0 = np.linspace(0, 1, n+1)
    t = np.linspace(0, 0.5, m+1)
    matrix[:, 0] = f(r0)
    matrix[0, :] = fi(t)
    matrix[n, :] = psi(t)
    for i in range(1, n):
        matrix[i, 1] = 0.5*(matrix[i+1, 0] + matrix[i-1, 0]) + h * F(r0[i])
    return matrix


def grid_method(u):
    """
    Реалізація методу сіток
    :param u: матриця
    :return:
    u - матриця після обчислення методу сіток
    """
    for j in range(1, m):
        for i in range(1, n):
            u[i, j + 1] = u[i + 1, j] + u[i - 1, j] - u[i, j - 1]
    return u


def print_grid(u):
    """
    Виводить матрицю з округленням до 4 знаків після коми
    :param u:
    :return:
    """
    u = np.round(u, 4)
    print(u)


def plot_3d_grid(u):
    """
    Малює матрицю в 3д просторі
    :param u:
    :return:
    """
    row0 = np.linspace(0, 1, n+1)
    t = np.linspace(0, 0.5, m+1)
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
u = u.T
print_grid(u)
plot_3d_grid(u)
