# Evaluate y = mx + c for given values
def linear_equation(m, c, x_values):
    return [m * x + c for x in x_values]

x_vals = list(range(-5, 6))
y_vals = linear_equation(2, 3, x_vals)

for x, y in zip(x_vals, y_vals):
    print(f"x={x}, y={y}")
