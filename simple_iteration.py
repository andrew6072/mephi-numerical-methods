import numpy as np

xmin = 0
xmax = 1.41
ymin = 0
ymax = 1
h = 0.01
tau = h**2 / 4

def f(x, y):
    return 70 * np.sin(5 * x + 2 * y) + (435 * x - 580 * y) * np.cos(5 * x + 2 * y) + 6


def phi1(y):
    return 5 * (-3 * xmin + 4 * y) * np.cos(5 * xmin + 2 * y) + 3 * y ** 2


def phi2(y):
    return 5 * (-3 * xmax + 4 * y) * np.cos(5 * xmax + 2 * y) + 3 * y ** 2


def psi1(x):
    return 5 * (-3 * x + 4 * ymin) * np.cos(5 * x + 2 * ymin) + 3 * ymin ** 2


def psi2(x):
    return 5 * (-3 * x + 4 * ymax) * np.cos(5 * x + 2 * ymax) + 3 * ymax ** 2


def solution(x, y):
    return 5 * (-3 * x + 4 * y) * np.cos(5 * x + 2 * y) + 3 * y ** 2


x = np.arange(xmin, xmax + h, h)
y = np.arange(ymin, ymax + h, h)

analytical_solution = np.zeros((len(y), len(x)))
for i, y_ in enumerate(y):
    for j, x_ in enumerate(x):
        analytical_solution[i][j] = solution(x_, y_)


numerical_solution = np.zeros((len(y), len(x)))
numerical_solution[0, :] = psi1(x)
numerical_solution[len(y)-1, :] = psi2(x)
numerical_solution[:, 0] = phi1(y)
numerical_solution[:, len(x)-1] = phi2(y)

print(np.linalg.norm(analytical_solution[0, :] - numerical_solution[0, :]))
print(np.linalg.norm(analytical_solution[len(y)-1, :] - numerical_solution[len(y)-1, :]))
print(np.linalg.norm(analytical_solution[:, 0] - numerical_solution[:, 0]))
print(np.linalg.norm(analytical_solution[:, len(x)-1] - numerical_solution[:, len(x)-1]))

def simple_iteration(x, y, h, tau, num_iteration):
    sol = np.zeros((len(y), len(x)))
    sol[0, :] = psi1(x)
    sol[len(y) - 1, :] = psi2(x)
    sol[:, 0] = phi1(y)
    sol[:, len(x) - 1] = phi2(y)
    new_sol = np.copy(sol)
    nx = len(x)
    ny = len(y)
    for _ in range(num_iteration):
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                new_sol[i, j] = sol[i, j] + tau * ( (sol[i+1, j] - 2 * sol[i, j] + sol[i-1, j]) / h**2 +
                                                    (sol[i, j+1] - 2 * sol[i, j] + sol[i, j-1]) / h**2 + f(x[j], y[i]))
        sol = np.copy(new_sol)
    return sol

numerical_solution = simple_iteration(x, y, h, tau, 100)
