from sympy import symbols, diff, solve, Matrix

# Define variables and function
x, y = symbols('x y')
f = x**2 + y**2

# Calculate first derivatives
f_x = diff(f, x)
f_y = diff(f, y)

# Find critical points
critical_points = solve([f_x, f_y], (x, y))

# Calculate second derivatives
f_xx = diff(f_x, x)
f_yy = diff(f_y, y)
f_xy = diff(f_x, y)

# Evaluate Hessian determinant
x_val, y_val = critical_points[x], critical_points[y]
Hessian = Matrix([[f_xx, f_xy], [f_xy, f_yy]])
det_H = Hessian.det()
det_H_value = det_H.subs({x: x_val, y: y_val})

if det_H_value > 0:
    if f_xx.subs({x: x_val, y: y_val}) > 0:
        print(f"Minimum at ({x_val}, {y_val})")
    else:
        print(f"Maximum at ({x_val}, {y_val})")
elif det_H_value < 0:
    print(f"Saddle point at ({x_val}, {y_val})")
else:
    print(f"Indeterminate at ({x_val}, {y_val})")
