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

X, Y = np.meshgrid(x, y)

analytical_solution = np.zeros((len(y), len(x)))
for i, y_ in enumerate(y):
    for j, x_ in enumerate(x):
        analytical_solution[i][j] = solution(x_, y_)


# numerical_solution = f(X, Y)
# numerical_solution[0, :] = psi1(x)
# numerical_solution[len(y)-1, :] = psi2(x)
# numerical_solution[:, 0] = phi1(y)
# numerical_solution[:, len(x)-1] = phi2(y)
#
# print(np.linalg.norm(analytical_solution[0, :] - numerical_solution[0, :]))
# print(np.linalg.norm(analytical_solution[len(y)-1, :] - numerical_solution[len(y)-1, :]))
# print(np.linalg.norm(analytical_solution[:, 0] - numerical_solution[:, 0]))
# print(np.linalg.norm(analytical_solution[:, len(x)-1] - numerical_solution[:, len(x)-1]))

def simple_iteration(x, y, h, tau, num_iteration):
    sol = f(X, Y)
    sol[0, :] = psi1(x)
    sol[len(y) - 1, :] = psi2(x)
    sol[:, 0] = phi1(y)
    sol[:, len(x) - 1] = phi2(y)
    new_sol = np.copy(sol)
    for _ in range(num_iteration):
        new_sol[1:-1, 1:-1] = sol[1:-1, 1:-1] + tau * ( (sol[2:, 1:-1] - 2 * sol[1:-1, 1:-1] + sol[0:-2, 1:-1]) / h**2 +
                                                        (sol[1:-1, 2:] - 2 * sol[1:-1, 1:-1] + sol[1:-1, 0:-2]) / h**2 + f(X[1:-1, 1:-1], Y[1:-1, 1:-1]))
        sol = np.copy(new_sol)
    return sol


numerical_solution = simple_iteration(x, y, h, tau, 10000)
print(np.linalg.norm(analytical_solution - numerical_solution))