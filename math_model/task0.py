import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, solve, Matrix


def func(x, y):
    return -2*(x**2) + 1.05*(x**4) - (x**6)/6 - x*y - y**2

def func_1(u1, u2):
    return -1.1*(u1**2) - 1.5*(u2**2) + 2*u1*u2 + u1 + 5


x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)

x, y = np.meshgrid(x, y)

z = func(x, y)

fig = plt.figure()

ax = fig.add_subplot(111)

contour = ax.contour(x, y, z, levels=50,)
plt.colorbar(contour)
extremum_1 = (0, 0)
trajectory_1 = [(-3, 2), (-4, 1), (0, 0), (4, 4), extremum_1]
trajectory_2 = [(-2, 3), (1, -4), (2, 2.25), (0, 1), extremum_1]


def traj(points, color, num):
    xs = [i[0] for i in points]
    ys = [i[1] for i in points]
    ax.plot(xs, ys, 'b-', color=color, label=f"Метод{num}({len(points)-1})")
    for ind, point in enumerate(points):
        plt.annotate(f"{point}", point, textcoords="offset points", xytext=(0, 10), ha='center')


traj(trajectory_1, 'red', 1)
traj(trajectory_2, 'green', 2)

plt.title("Лінії рівня функції f(x, y)=-2x^2+1.05x^4-x^6/6-xy-y^2")

ax.plot(extremum_1[0], extremum_1[1], 'X', label=f"Максимум{extremum_1}")
plt.legend(loc="upper left")
ax.set_xlabel('U2')
ax.set_ylabel('U1')
plt.text(16, 0, 'значення f(z)', rotation=90, verticalalignment='center')
plt.show()
