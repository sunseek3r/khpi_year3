import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.fft import fft
import matplotlib.pyplot as plt
def pend(t, f, a,b,c):
    x, y, z = f
    dydt = [a*(y - x), x*(b - z) - y, x*y - c*z]
    return dydt
def func(x, a,b,c):
    return [x[1], c*(a-x[0]**2)*x[1] - b*x[0]]
a = [10]
b = [30]
c = 1.7
y0 = [10, 10, 10] #fsolve(func, [100000, 1000], args=(e, g))
#ycr = fsolve(func, [5, 5], args=(a,b,c))

t1, t2 = 0, 1800
t = np.linspace(t1, t2, 160000)

for j in b:
    for i in a:
        fig, ax = plt.subplots(2,2, figsize=(10,10))
        ax[0,0].remove()
        ax[0,0] = fig.add_subplot(2, 2, 1, projection='3d')
        sol = solve_ivp(pend, [t1, t2], y0, t_eval=t, args=(i,j,c))
        ax[0,0].plot(sol.y[0], sol.y[1], sol.y[2], lw = 0.5)
        ax[0,1].plot(t, sol.y[0], color = "red")
        ax[1,0].plot(t, sol.y[1], color = "orange")
        ax[1,1].plot(t, sol.y[2], color = "green")

        ax[0, 0].set_title(f"Траекторія системи\n\n σ = {i} r = {j} b = {c} x0 = {y0}")
        ax[0, 0].set_xlabel("X")
        ax[0, 0].set_ylabel("Y")
        ax[0, 0].set_zlabel("Z")
        ax[0, 1].set_title("Залежність X від часу")
        ax[1, 0].set_title("Залежність Y від часу")
        ax[1, 1].set_title("Залежність Z від часу")
        ax[0, 1].set_xlabel("Час, t")
        ax[0, 1].set_ylabel("X(t)")
        ax[1, 0].set_xlabel("Час, t")
        ax[1, 0].set_ylabel("Y(t)")
        ax[1, 1].set_xlabel("Час, t")
        ax[1, 1].set_ylabel("Z(t)")
        plt.savefig(f'a{i}b{j}c{c}.png', bbox_inches='tight')
        plt.show()