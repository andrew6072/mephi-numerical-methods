import numpy as np
import matplotlib.pyplot as plt

# Размеры области
xmin, xmax = 0.0, 1.0
ymin, ymax = 0.0, 1.4

# Точное решение
def u0(x, y):
    return np.sin(7.0*x - 3.0*y) + 2.0*np.cos(2.0*x + 6.0*y)

# Неоднородность в уравнении
def f(x, y):
    return 58.0*np.sin(7.0*x - 3.0*y) + 80.0*np.cos(2.0*x + 6.0*y)

# Граничные условия
def phi_L(y):
    return -7.0*np.cos(7.0*xmin - 3.0*y) + 4.0*np.sin(2.0*xmin + 6.0*y)

def phi_R(y):
    return +7.0*np.cos(7.0*xmax - 3.0*y) - 4.0*np.sin(2.0*xmax + 6.0*y)

def phi_B(x):
    return +3.0*np.cos(7.0*x - 3.0*ymin) + 12.0*np.sin(2.0*x + 6.0*ymin)

def phi_T(x):
    return -3.0*np.cos(7.0*x - 3.0*ymax) - 12.0*np.sin(2.0*x + 6.0*ymax)


# Характерное разбиение
h0 = 0.01

nx = int(np.round((xmax - xmin) / h0))
ny = int(np.round((ymax - ymin) / h0))

hx = (xmax - xmin) / nx
hy = (ymax - ymin) / ny

# Сетка (центры ячеек)
X, Y = np.meshgrid(
    np.linspace(xmin + 0.5*hx, xmax - 0.5*hx, nx),
    np.linspace(ymin + 0.5*hy, ymax - 0.5*hy, ny),
    indexing='ij'
)

# Подействовать оператором на матрицу
def OpA(u):
    res = np.zeros_like(u)
    res[+1:, :] += (u[+1:, :] - u[:-1, :]) * (hy / hx)
    res[:-1, :] += (u[:-1, :] - u[+1:, :]) * (hy / hx)
    res[:, +1:] += (u[:, +1:] - u[:, :-1]) * (hx / hy)
    res[:, :-1] += (u[:, :-1] - u[:, +1:]) * (hx / hy)
    return res

# Матрица правых частей
def RHS(x, y):
    res = f(x, y) * hx * hy
    res[ 0, :] += phi_L(y[ 0, :]) * hy
    res[-1, :] += phi_R(y[-1, :]) * hy
    res[:,  0] += phi_B(x[:,  0]) * hx
    res[:, -1] += phi_T(x[:, -1]) * hx
    return res


# Нормировка для однозначного определения решения задачи
# Неймана, среднее значение u(x, y) = 0.
u_exact = u0(X, Y) - np.mean(u0(X, Y))

# Sum(F) = 0. Условие существования решения
# для задачи Неймана.
F = RHS(X, Y)
F -= np.mean(F)


u = RHS(X, Y)

errors = []

err = 1.0
counter = 0

# Оценка погрешности конечно-объемной аппроксимации err ~ h
while err > 1.0e-3 * h0:
    rk = OpA(u) - F
    Ark = OpA(rk)
    tau = np.sum(Ark * rk) / np.sum(Ark * Ark)
    u -= tau * rk
    err = tau * np.linalg.norm(rk)

    # Нормировка для однозначного решения задачи
    # Неймана, среднее значение = 0.
    u -= np.mean(u)

    # Счетчик для отслеживания сходимости
    counter += 1
    if counter % 1000 == 0:
        print('  Step %d, error: %e' % (counter, err))

    # Массив u_exact не учавствует в решении
    # error для построения графика
    errors.append(np.linalg.norm(u - u_exact))


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

img2 = ax2.imshow(np.abs(u - u_exact).T, **options)
fig.colorbar(img2, ax=ax2)

fig.tight_layout()

plt.show()


# Проверим теорию, построим зависимость error(step)
lx = xmax - xmin
ly = ymax - ymin

Lmin = (2.0 * np.sin(0.5 * np.pi * hx / lx) / hx)**2 + (2.0 * np.sin(0.5 * np.pi * hy / ly) / hy)**2
Lmax = (2.0 * np.cos(0.5 * np.pi * hx / lx) / hx)**2 + (2.0 * np.cos(0.5 * np.pi * hy / ly) / hy)**2
x_t = Lmin / Lmax
q_t = (1 - x_t)/(1 + x_t)


# y = k x + b
def lstsq(x, y):
    n = len(x)

    k = (n *np.sum(x*y) - np.sum(x)*np.sum(y)) / (n * np.sum(x*x) - np.sum(x)**2)
    b = np.mean(y) - k * np.mean(x)
    return k, b


all_steps = np.arange(0, len(errors))

beg =  5000
end = 45000

steps = np.array(all_steps[beg:end])
errs = np.array(errors[beg:end])

k, b = lstsq(steps, np.log(errs))
q_p = np.exp(k)
C = np.exp(b)

x_p = (1 - q_p) / (1 + q_p)

fig = plt.figure(dpi=200, figsize=(6.0, 4.0))

ax = plt.gca()

ax.set_title('Погрешность от итерации')

ax.semilogy(all_steps, errors, color='green')
ax.semilogy(all_steps, C*np.power(q_p, all_steps), linestyle='dashed', color='black', linewidth=1.0)

text = '$\\xi_{теор} = $ % .2e\n' % x_t
text += '$\\rho_{теор} = $ % .7f\n\n' % q_t
text += '$\\xi_{прак} = $ % .2e\n' % x_p
text += '$\\rho_{прак} = $ % .7f\n' % q_p

ax.text(0.0, np.min(errors), text)

plt.show()
