import numpy as np

# Constants
m = 10        # kg
gamma = 0.04  # N/(m/s)
v0 = 126      # m/s
g = 9.8       # m/s^2
L = 1480      # target distance
vt = m * g / gamma  # terminal velocity

# Functions for motion
def x_t(t, theta):
    return (m * v0 * np.cos(theta) / gamma) * (1 - np.exp(-gamma * t / m))

def y_t(t, theta):
    return (vt / g) * (v0 * np.sin(theta) + vt) * (1 - np.exp(-g * t / vt)) - vt * t

# Solve for flight time given theta (numerical solve for x(t) = L)
def flight_time(theta):
    # Search for root in time (0 to 100 s should be enough)
    t_min, t_max = 0, 100
    for _ in range(200):
        t_mid = 0.5 * (t_min + t_max)
        if x_t(t_mid, theta) < L:
            t_min = t_mid
        else:
            t_max = t_mid
    return 0.5 * (t_min + t_max)

# Root function in theta
def f_theta(theta):
    t_star = flight_time(theta)
    return y_t(t_star, theta)

# Bisection method for theta root finding
def find_root(function, a, b):
    iter = 0
    f_a = function(a)
    f_b = function(b)
    root = None
    for _ in range(100):
        iter += 1
        c = (a + b) / 2
        f_c = function(c)

        if np.isclose(f_c, 0, 1.0E-6):
            root = c
            break
        elif f_a * f_c < 0:
            b = c
            f_b = f_c
        else:
            a = c
            f_a = f_c
        root = c
    return root, iter

theta_root, iterations = find_root(f_theta, 0.01, np.pi/2 - 0.01)  # in radians
t_hit = flight_time(theta_root)

print(f"Cannon inclination angle = {np.degrees(theta_root):.2f} degree")
print(f"Flight time = {t_hit:.2f} s")

