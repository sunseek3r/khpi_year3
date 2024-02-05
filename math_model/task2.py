import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve

# Функція для розв'язання системи та побудови графіків
def solve_and_plot(e1, gamma12, gamma11, gamma21, e2, gamma22, x0, y0):
    # Оголошення системи рівнянь
    x, y = symbols('x y')
    eq1 = Eq(e1 * x - gamma12 * x * y - gamma11 * x**2, 0)
    eq2 = Eq(gamma21 * x * y - e2 * y - gamma22 * y**2, 0)

    # Розв'язуємо систему рівнянь для стаціонарних точок
    stationary_points = solve((eq1, eq2), (x, y))

    x_values_steady = []
    y_values_steady = []

    print("Стаціонарні точки:")
    for point in stationary_points:
        x_val, y_val = point
        x_values_steady.append(x_val)
        y_values_steady.append(y_val)
        print(f"x = {x_val}, y = {y_val}")

    # Оголошення системи рівнянь для чисельного розв'язку
    def system(t, variables):
        x, y = variables
        dxdt = e1 * x - gamma12 * x * y - gamma11 * x**2
        dydt = gamma21 * x * y - e2 * y - gamma22 * y**2
        return [dxdt, dydt]

    # Часовий інтервал для розв'язку
    t_span = (0, 30)

    # Розв'язок системи рівнянь
    solution = solve_ivp(system, t_span, [x0, y0], t_eval=np.linspace(t_span[0], t_span[1], 1000))

    # Отримання результатів
    t = solution.t
    x_values = solution.y[0]
    y_values = solution.y[1]

    # Побудова графіків у часі
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, x_values, label='Кількість жертв (x)')
    plt.plot(t, y_values, label=f'Кількість хижаків (y):')
    plt.xlabel('Час')
    plt.ylabel('Популяція')
    plt.title(f'Залежність популяцій від часу e1={e1}, gamma12={gamma12}, gamma11={gamma11}, gamma21={gamma21}, e2={e2}, gamma22={gamma22}, x0={x0}, y0={y0}')
    plt.legend()
    plt.grid(True)

    # Побудова графіку у фазовому просторі
    plt.subplot(2, 1, 2)
    plt.plot(x_values, y_values, label='Фазовий простір')
    plt.scatter(x_values_steady[2], y_values_steady[2], color='red', label=f'Стаціонарні точки {round(x_values_steady[2], 3)} {round(y_values_steady[2],3)}')
    plt.xlabel('Кількість жертв (x)')
    plt.ylabel('Кількість хижаків (y)')
    plt.title('Фазовий простір')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Значення параметрів моделі
params = [
    #e1, gamma12, gamma11, gamma21, e2, gamma22, x0, y0):
#    (2.0, 0.5, 0.1, 0.7, 3.0, 0.11),


#    (0.9, 0.5, 0.1, 0.7, 3., 0.11),
#    (1.9, 0.5, 0.1, 0.7, 3., 0.11),
#    (2.9, 0.5, 0.1, 0.7, 3., 0.11),
#    (3.9, 0.5, 0.1, 0.7, 3., 0.11),

#    (2.0, 0.5, 0.1, 0.7, 3., 0.11),
#    (2.0, 1.5, 0.1, 0.7, 3., 0.11),
#    (2.0, 2.5, 0.1, 0.7, 3., 0.11),
#    (2.0, 3.5, 0.1, 0.7, 3., 0.11),

#    (2.0, 0.5, 0.1, 0.7, 3., 0.11),
#    (2.0, 0.5, 1.1, 0.7, 3., 0.11),
#    (2.0, 0.5, 2.1, 0.7, 3., 0.11),
#    (2.0, 0.5, 3.1, 0.7, 3., 0.11),

#    (2.0, 0.5, 0.1, 0.7, 3., 0.11),
#    (2.0, 0.5, 0.1, 1.7, 3., 0.11),
#    (2.0, 0.5, 0.1, 2.7, 3., 0.11),
#    (2.0, 0.5, 0.1, 3.7, 3., 0.11),

#    (2.0, 0.5, 0.1, 0.7, 3., 0.11),
#    (2.0, 0.5, 0.1, 0.7, 4., 0.11),
#    (2.0, 0.5, 0.1, 0.7, 5., 0.11),
#    (2.0, 0.5, 0.1, 0.7, 6., 0.11),

    (2.0, 0.5, 0.1, 0.7, 3., 0.11),
    (2.0, 0.5, 0.1, 0.7, 3., 1.11),
    (2.0, 0.5, 0.1, 0.7, 3., 2.11),
    (2.0, 0.5, 0.1, 0.7, 3., 3.11),

]

# Початкові умови
x0 = 2
y0 = 2.7

# Моделювання та побудова графіків для різних параметрів
for param_set in params:
    solve_and_plot(*param_set, x0, y0)