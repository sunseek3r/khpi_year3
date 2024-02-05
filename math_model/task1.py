import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Оголосимо символи
x, y = sp.symbols('x y')

# Введемо оновлені параметри моделі
epsilon = 0.8  # ε
beta = 0.45 # β
alpha = 0.014 # α
delta = 0.005 # δ

# Визначимо рівняння моделі
dx_dt = epsilon * x - alpha * x * y
dy_dt = delta * x * y - beta * y

# Знайдемо стаціонарні точки, де dx/dt = 0 і dy/dt = 0
stationary_points = sp.solve([dx_dt, dy_dt], (x, y))

print("Стаціонарні точки:")
for point in stationary_points:
    print(f"x_st = {point[0]}, y_st = {point[1]}")


# Оголосимо функцію, яка представляє систему рівнянь для чисельного розв'язку
def model(t, z):
    x, y = z
    dx_dt = (epsilon - alpha) * x - alpha * x * y
    dy_dt = delta * x * y - beta * y
    return [dx_dt, dy_dt]


# Встановимо часовий інтервал
t_start = 0
t_end = 100
t_step = 0.01
t_span = (t_start, t_end)

# Початкові умови для чисельного розв'язку
initial_conditions_list = [
    [1, 1],
]


# Побудуємо графіки


plt.figure(figsize=(10, 8))
for initial_conditions in initial_conditions_list:
    solution = solve_ivp(model, t_span, initial_conditions, t_eval=np.arange(t_start, t_end, t_step))
    t = solution.t
    x = solution.y[0]
    y = solution.y[1]
    plt.plot(t, x, label=f'Зайці: {initial_conditions[0]}, Вовки: {initial_conditions[1]}')

plt.xlabel('Час')
plt.ylabel('Популяція')
plt.legend()
plt.title(f'epsilon = {epsilon} beta = {beta} alpha = {alpha} delta ={delta}')
plt.grid(True)

#-----------------------------------------------------------------------------------

initial_conditions_phase = [
    [1, 1],

]

plt.figure(figsize=(12, 6))

# Побудуємо фазові траєкторії
for ic in initial_conditions_phase:
    phase_solution = solve_ivp(model, t_span, ic, t_eval=np.arange(t_start, t_end, t_step))
    plt.plot(phase_solution.y[0], phase_solution.y[1], label=f'ПУ={ic}', alpha=0.7)

# Додамо стаціонарні точки на графік фазових траєкторій
x_st_values = [point[0] for point in stationary_points]
y_st_values = [point[1] for point in stationary_points]

plt.scatter(x_st_values, y_st_values, color='red', label=f'Стаціонарні точки {x_st_values} {y_st_values}')

plt.xlabel('Кількість жертв (x)')
plt.ylabel('Кількість хижаків (y)')
plt.title(f'Фазові траєкторії x0={initial_conditions_phase[0][0]} y0={initial_conditions_phase[0][1]}')
plt.legend()
plt.grid(True)
#-----------------------------------------------------------------------------------

# Знайдемо максимальну та мінімальну кількості хижаків та жертв
max_x = np.max(x)
min_x = np.min(x)
max_y = np.max(y)
min_y = np.min(y)

print(f"Максимальна кількість жертв: {max_x}")
print(f"Мінімальна кількість жертв: {min_x}")
print(f"Максимальна кількість хижаків: {max_y}")
print(f"Мінімальна кількість хижаків: {min_y}")

#-----------------------------------------------------------------------------------
peaks_y, _ = find_peaks(y)

# Знайдемо періоди між піками
periods_y = np.diff(t[peaks_y])

# Знайдемо мінімуми у графіку кількості хижаків
minima_y, _ = find_peaks(-y)

# Знайдемо періоди між мінімумами
periods_minima_y = np.diff(t[minima_y])

# Виведемо знайдені періоди
print("Періоди коливань чисельності хижаків:")
print(periods_y)
print("Періоди коливань чисельності хижаків (за мінімумами):")
print(periods_minima_y)

# Знайдемо піки (максимуми) у графіку кількості жертв
peaks_x, _ = find_peaks(x)

# Знайдемо періоди між піками
periods_x = np.diff(t[peaks_x])

# Знайдемо мінімуми у графіку кількості жертв
minima_x, _ = find_peaks(-x)

# Знайдемо періоди між мінімумами
periods_minima_x = np.diff(t[minima_x])

# Виведемо знайдені періоди
print("Періоди коливань чисельності жертв:")
print(periods_x)
print("Періоди коливань чисельності жертв (за мінімумами):")
print(periods_minima_x)
#-----------------------------------------------------------------------------------

plt.show()
