import matplotlib.pyplot as plt
import numpy as np
import numdifftools as nd
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp

np.set_printoptions(precision=5, suppress=True)


def economic_model(xarr):
    τ, σ = xarr
    α = 0.5
    β = 1.5
    γ = 1.5
    δ = 0.1
    ν = 5
    μ = 20
    λ = 20
    ρ = 10
    A0 = 1
    L0 = 1
    D0 = 1

    def pend(t, f):
        Π1, p1, w1, K1, Π2, p2, w2, K2 = f

        Θ = (1 + α * (β - 1)) ** -1
        T = τ * Π1
        r = np.exp(-ρ * σ * T)

        L1 = K1 * (((1 - α) * A0 * p1 / w1) ** (1 / α))
        D1 = D0 * np.exp(-β * p1) * p2 / (p1 + p2)
        S1 = L0 * (1 - np.exp(-γ * w1)) * w1 / (w1 + w2)
        Q1 = A0 * (K1 ** α) * (L1 ** (1 - α))
        I1 = (1 - τ) * (1 - Θ) * Π1

        L2 = K2 * (((1 - α) * A0 * p2 / w2) ** (1 / α))
        D2 = D0 * np.exp(-β * p2) * p1 / (p1 + p2)
        S2 = L0 * (1 - np.exp(-γ * w2)) * w2 / (w1 + w2)
        Q2 = A0 * (K2 ** α) * (L2 ** (1 - α))
        I2 = (1 - Θ) * Π2

        _Q1 = np.min([Q1, D1])
        _L1 = np.min([L1, S1])

        _Q2 = np.min([Q2, D2])
        _L2 = np.min([L2, S2])

        dydt = [(p1 * _Q1 - w1 * _L1 - Π1) / ν,
                (D1 - Q1) / μ,
                (L1 - S1) / λ,
                -δ * K1 + I1,
                (r * p2 * _Q2 - w2 * _L2 - Π2) / ν,
                (D2 - Q2) / μ,
                (L2 - S2) / λ,
                -δ * K2 + I2]

        return dydt

    y0 = [0, 0.5, 0.25, 0.1, 0, 0.5, 0.25, 0.1]

    t1, t2 = 0, 600
    t = np.linspace(t1, t2, 1000)

    sol = solve_ivp(pend, [t1, t2], y0, t_eval=t)
    Π1, p1, w1, K1, Π2, p2, w2, K2 = sol.y
    L1 = K1 * (((1 - α) * A0 * p1 / w1) ** (1 / α))
    Q1 = A0 * (K1 ** α) * (L1 ** (1 - α))

    L2 = K2 * (((1 - α) * A0 * p2 / w2) ** (1 / α))
    Q2 = A0 * (K2 ** α) * (L2 ** (1 - α))
    return Q2[-1] / Q1[-1]


def gold_section(a, b, f, ε):
    L = b - a
    x1 = a + 0.382 * L
    x2 = a + 0.618 * L
    f1, f2 = f(x1), f(x2)
    while 1:
        if f1 <= f2:
            b = x2
            L = b - a
            x2 = x1
            f2 = f1
            x1 = a + 0.382 * L
            f1 = f(x1)
        else:
            a = x1
            L = b - a
            x1 = x2
            f1 = f2
            x2 = a + 0.618 * L
            f2 = f(x2)

        if L <= ε:
            if f1 < f2:
                return x1
            else:
                return x2


def fast_descent_method(f, x0):
    x = x0
    logs = [x]
    while 1:
        # print(x, f(x))
        dk = -nd.Gradient(f)(x)
        # dk = dk/np.linalg.norm(dk)
        # print(dk)
        # plt.plot(fa)
        # plt.show()
        # arg = minimize_scalar(lambda alpha: f(x+alpha*dk), method='golden')
        # alpha = arg.x

        alpha = gold_section(0, 100, lambda alpha: f(x + alpha * dk), 0.000001)
        # print(arg.fun, arg.x)
        x = x + alpha * dk
        if np.linalg.norm(alpha * dk) < 0.01: return x, logs
        logs.append(x)
        # print(np.linalg.norm(alpha*dk))


def rotate(A, teta):
    B = np.array([[np.cos(teta), -np.sin(teta)],
                  [np.sin(teta), np.cos(teta)]])
    return B @ A


teta = 0


def fkv(xarr):
    A = np.array([[100, 0],
                  [0, 1]])
    b = np.array([40, 50])
    x0 = (A[0, 1] * b[1] - b[0] * A[1, 1]) / (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]) / 2
    y0 = (A[1, 0] * b[0] - b[1] * A[0, 0]) / (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]) / 2
    # print(x0,y0)

    X = xarr.copy()
    x, y = X
    # print(x)
    global teta
    if isinstance(x, np.ndarray):
        print("zaebumba")
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j], y[i, j] = rotate(np.array([x[i, j], y[i, j]]) - np.array([x0, y0]), teta) + [x0, y0]
    else:
        x, y = rotate(np.array([x, y]) - np.array([x0, y0]), teta) + [x0, y0]
    # print(x)
    # print(A)
    return A[0, 0] * x ** 2 + (A[0, 1] + A[1, 0]) * x * y + A[1, 1] * y ** 2 + b[0] * x + b[1] * y


make_complex = np.vectorize(complex)


def f(xarr):
    a, b = xarr
    g = make_complex(a, b)
    return -1/(1 + np.abs(g**6 - 1))


def plot_contour(f, k, xmin, xmax, ymin, ymax, fig, ax):
    x = np.arange(xmin, xmax, 0.01)
    y = np.arange(ymin, ymax, 0.01)
    xgrid, ygrid = np.meshgrid(x, y)
    xgrid = xgrid
    ygrid = ygrid
    z = -f(np.array([xgrid, ygrid]))

    cs = ax.contourf(xgrid, ygrid, z, levels=k)
    # cs.clabel(colors = 'k')
    fig.colorbar(cs)


def plot_path(data, label1, color, ax):
    data = np.round(data, 2)
    x, y = zip(*data)

    print(data)
    ax.plot(x, y, label=label1 + f'({len(data)} points)', color=color, marker="o")
    for i in range(len(data)):
        ax.text(x[i], y[i], f"({x[i]}, {y[i]})", ha='right', va='bottom')
    ax.legend()


def plot_point(point, color, label, ax):
    ax.scatter(*point, c=color, s=100, marker='x', label=label, zorder=1000)
    ax.legend()


def find_extremum(f):
    fig, ax = plt.subplots()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal')
    ax.set_title("f(x) = -20 + (10 * cos(2 * pi * x) - x**2) +  (10 * cos(2 * pi * y) - y**2)\n")

    plot_contour(f, 20, -2, 2, -2, 2, fig, ax)
    start_points = [[0, 1], [1, 0], [-1, 0]]
    for y0 in start_points:
        extremum, path = fast_descent_method(f, y0)
        extremum = np.round(extremum, 2)
        # print(extremum, np.round(f(extremum), 4))
        plot_path(path, f"Метод найшвидшого спуску(Початкова точка {y0})", "r", ax)
    plot_point(extremum, "blue", f"Точка мінімуму({extremum[0]}, {extremum[1]})", ax)
    plt.show()


# find_extremum(fkv)

def stat(f):
    fig, ax = plt.subplots()
    ax.set_ylabel("Кількість ітерацій")
    ax.set_xlabel("Кут повороту, θ")
    ax.set_title("Еліптичність ε = 100\n")
    global teta
    T = np.linspace(0, np.pi / 2, 1000)
    ans = []
    for t in T:
        teta = t
        extremum, path = fast_descent_method(f, [35, 27])
        ans.append(len(path) - 1)
    ax.plot(T, ans)
    ax.set_xticks([0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])
    ax.set_xticklabels(['0', r'$\frac{\pi}{6}$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{3}$', r'$\frac{\pi}{2}$'])
    plt.show()


# print(f(xarr=[-1, 0]))


stat(fkv)
find_extremum(f)
