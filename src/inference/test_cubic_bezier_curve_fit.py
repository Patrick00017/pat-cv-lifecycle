import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def cubic_bezier(t, px0, py0, px1, py1, px2, py2, px3, py3):
    u = 1 - t
    tt = t * t
    uu = u * u
    uuu = uu * u
    ttt = tt * t
    
    x = uuu * px0 + 3 * uu * t * px1 + 3 * u * tt * px2 + ttt * px3
    y = uuu * py0 + 3 * uu * t * py1 + 3 * u * tt * py2 + ttt * py3
    return np.column_stack([x, y])


def bezier_x(t, px0, px1, px2, px3):
    u = 1 - t
    return u**3 * px0 + 3 * u**2 * t * px1 + 3 * u * t**2 * px2 + t**3 * px3


def bezier_y(t, py0, py1, py2, py3):
    u = 1 - t
    return u**3 * py0 + 3 * u**2 * t * py1 + 3 * u * t**2 * py2 + t**3 * py3


np.random.seed(42)
num_points = 20
t_data = np.linspace(0, 1, num_points)

px0, py0 = 0.0, 0.0
px3, py3 = 1.0, 1.0
px1, py1 = 0.5, 0.8
px2, py2 = 0.8, 0.2

x_data = bezier_x(t_data, px0, px1, px2, px3)
y_data = bezier_y(t_data, py0, py1, py2, py3)

x_data += np.random.normal(0, 0.05, num_points)
y_data += np.random.normal(0, 0.05, num_points)

popt_x, _ = curve_fit(bezier_x, t_data, x_data, p0=[0.0, 0.5, 0.8, 1.0])
popt_y, _ = curve_fit(bezier_y, t_data, y_data, p0=[0.0, 0.8, 0.2, 1.0])

t_fit = np.linspace(0, 1, 100)
x_fit = bezier_x(t_fit, *popt_x)
y_fit = bezier_y(t_fit, *popt_y)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(t_data, x_data, c='blue', label='Sampled x')
plt.plot(t_fit, x_fit, 'r-', label='Fitted x')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(t_data, y_data, c='blue', label='Sampled y')
plt.plot(t_fit, y_fit, 'r-', label='Fitted y')
plt.legend()

plt.tight_layout()
plt.savefig('curve_fit_result.png')
plt.show()

print(f"Fitted control points X: {popt_x}")
print(f"Fitted control points Y: {popt_y}")
