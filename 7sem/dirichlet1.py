import numpy as np
import matplotlib.pyplot as plt

# A_i x_{i-1} + B_i x_i + C_i x_{i+1} = F_i
# A_i, B_i, C_i, F_i могут быть векторами --- задача решается
# одновременно для набора трехдиагональных матриц
def progon(A, B, C, F):
    n = F.shape[0]
    x = np.zeros_like(F)
    alfa = np.zeros_like(F)
    beta = np.zeros_like(F)

    alfa[0] = - C[0] / B[0]
    beta[0] =   F[0] / B[0]

    for i in range(1, n):
        xi = 1.0 / (B[i] + A[i] * alfa[i - 1])
        alfa[i] = -C[i] * xi
        beta[i] = (F[i] - A[i] * beta[i - 1]) * xi

    x[-1] = beta[-1]
    for i in range(n - 2, -1, -1):
        x[i] = alfa[i] * x[i + 1] + beta[i]

    return x


# Размеры области
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, np.sqrt(2.0)

# Точное решение
def u_exact(x, y):
    return np.sin(7.0*x - 3.0*y) + 2.0*np.cos(2.0*x + 6.0*y)

# Неоднородность в уравнении
def f(x, y):
    return 58.0*np.sin(7.0*x - 3.0*y) + 80.0*np.cos(2.0*x + 6.0*y)

# Граничные условия
def phi_L(y):
    return u_exact(xmin, y)

def phi_R(y):
    return u_exact(xmax, y)

def phi_B(x):
    return u_exact(x, ymin)

def phi_T(x):
    return u_exact(x, ymax)


# Внутренние узлы
def inner(u):
    return u[1:-1, 1:-1]

# Вторая производная по x во внутренних узлах
def diff_X2(u):
    return u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]

def diff_Y2(u):
    return u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]


# Характерное разбиение
h0 = 0.01

lx = xmax - xmin
ly = ymax - ymin

nx = int(np.round(lx / h0)) + 1
ny = int(np.round(ly / h0)) + 1

hx = lx / (nx - 1)
hy = ly / (ny - 1)

Lmin = (2.0 * np.sin(0.5 * np.pi * hx / lx) / hx)**2 + (2.0 * np.sin(0.5 * np.pi * hy / ly) / hy)**2
Lmax = (2.0 * np.cos(0.5 * np.pi * hx / lx) / hx)**2 + (2.0 * np.cos(0.5 * np.pi * hy / ly) / hy)**2

# Оптимальный параметр прогонки
tau = 1.0 / np.sqrt(Lmin * Lmax)

tau_x = tau / (hx * hx)
tau_y = tau / (hy * hy)

# Сетка (центры ячеек)
X, Y = np.meshgrid(
    np.linspace(xmin, xmax, nx),
    np.linspace(ymin, ymax, ny),
    indexing='ij'
)

F = f(X, Y)
u0 = u_exact(X, Y)

u = np.zeros_like(X)
u[ 0, :] = phi_L(Y[ 0, :])
u[-1, :] = phi_R(Y[-1, :])
u[:,  0] = phi_B(X[:,  0])
u[:, -1] = phi_T(X[:, -1])

# Матрицы для прогонки
Fy = np.zeros(ny)
Ay = -tau_y * np.ones(ny)
By = 1.0 + 2.0 * tau_y * np.ones(ny)
Cy = -tau_y * np.ones(ny)
Ay[0] = Ay[-1] = 0.0
By[0] = By[-1] = 1.0
Cy[0] = Cy[-1] = 0.0

Fx = np.zeros(nx)
Ax = -tau_x * np.ones(nx)
Bx = 1.0 + 2.0 * tau_x * np.ones(nx)
Cx = -tau_x * np.ones(nx)
Ax[0] = Ax[-1] = 0.0
Bx[0] = Bx[-1] = 1.0
Cx[0] = Cx[-1] = 0.0

errors = []

err = 1.0
counter = 0

# Оценка невязки для аппроксимации err ~ 4 h^2
while err > 4.0 * h0**2:
    # Прогонка в одну сторону
    uh = np.copy(u)
    for i in range(1, nx - 1):
        Fy[0] = u[i, 0]
        Fy[1:-1] = u[i, 1:-1] + tau * (u[i-1, 1:-1] - 2.0*u[i, 1:-1] + u[i+1, 1:-1]) / (hx*hx) + tau * F[i, 1:-1]
        Fy[-1] = u[i, -1]

        uh[i, :] = progon(Ay, By, Cy, Fy)

    # Прогонка в другую сторону
    un = np.copy(u)
    for j in range(1, ny - 1):
        Fx[0] = u[0, j]
        Fx[1:-1] = uh[1:-1, j] + tau * (uh[1:-1, j-1] - 2.0*uh[1:-1, j] + uh[1:-1, j+1]) / (hy*hy) + tau * F[1:-1, j]
        Fx[-1] = u[-1, j]

        un[:, j] = progon(Ax, Bx, Cx, Fx)

    # Проверка уравнений
    #delta1 = inner(uh) - inner(u) - tau_x * diff_X2(u) - tau_y * diff_Y2(uh) - tau * inner(F)
    #print('delta1: ', np.max(np.abs(delta1)))
    #delta2 = inner(un) - inner(uh) - tau_x * diff_X2(un) - tau_y * diff_Y2(uh) - tau * inner(F)
    #print('delta2: ', np.max(np.abs(delta2)))

    u = un

    residual = diff_X2(u) / (hx * hx) + diff_Y2(u) / (hy * hy) + inner(F)
    err = np.max(np.abs(residual))

    # Счетчик для отслеживания сходимости
    counter += 1
    if counter % 20 == 0:
        q = errors[-1]/errors[-2]
        print('  Step %d, error: %.2e, q: %.5f' % (counter, err, q))

        # Достигнута минимальная погрешность по решению
        if errors[-1] > errors[-9]:
            break

    # Массив u_exact не учавствует в решении
    # errors для построения графика
    errors.append(np.max(np.abs(u - u0)))


fig = plt.figure(dpi=200, figsize=(8.0, 4.0))

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

ax1.set_aspect('equal')
ax1.set_title('Решение')

ax2.set_aspect('equal')
ax2.set_title('Погрешность')

options = { 'cmap': 'jet',
            'origin': 'lower',
            'extent': [xmin, xmax, ymin, ymax],
            'interpolation': 'bilinear'}

img1 = ax1.imshow(u.T, **options)
fig.colorbar(img1, ax=ax1)

img2 = ax2.imshow(np.abs(u - u0).T, **options)
fig.colorbar(img2, ax=ax2, format='%.2e')

fig.tight_layout()

plt.show()


# Построим зависимость error(step)
fig = plt.figure(dpi=200, figsize=(6.0, 4.0))
ax = plt.gca()
ax.set_title('Погрешность от итерации')
ax.set_xlabel('Номер итерации')
ax.set_ylabel('Погрешность')
ax.semilogy(errors, color='black')
plt.show()
