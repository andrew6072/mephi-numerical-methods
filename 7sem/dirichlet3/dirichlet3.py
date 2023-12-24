import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg
import time


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


# Характерное разбиение
h0 = 0.002
lx = xmax - xmin
ly = ymax - ymin

nx = int(np.round(lx / h0)) + 1
ny = int(np.round(ly / h0)) + 1

beg_1 = time.time()

# Узлы сетки
x, hx = np.linspace(xmin, xmax, nx, retstep=True)
y, hy = np.linspace(ymin, ymax, ny, retstep=True)
X, Y = np.meshgrid(x, y, indexing='ij')

N = np.arange(0, X.size, dtype=int).reshape((nx, ny))

# Это очень интересный феномен, при задании cx = -1.0 / hx**2
# Не достигается второй порядок точности, почему?
cx = -((nx - 1) / lx)**2
cy = -((ny - 1) / ly)**2
bb = -2.0 * (cx + cy)

R1 = N[1:-1, 1:-1].flatten()
C1 = N[1:-1, 1:-1].flatten()
V1 = np.full_like(R1, bb)

R2 = N[1:-1, 1:-1].flatten()
C2 = N[0:-2, 1:-1].flatten()
V2 = np.full_like(R2, cx)

R3 = N[1:-1, 1:-1].flatten()
C3 = N[2:, 1:-1].flatten()
V3 = np.full_like(R3, cx)

R4 = N[1:-1, 1:-1].flatten()
C4 = N[1:-1, 0:-2].flatten()
V4 = np.full_like(R4, cy)

R5 = N[1:-1, 1:-1].flatten()
C5 = N[1:-1, 2:].flatten()
V5 = np.full_like(R5, cy)

# Граничные условия
R6 = N[0, :]
C6 = N[0, :]
V6 = np.ones_like(R6)

R7 = N[-1, :]
C7 = N[-1, :]
V7 = np.ones_like(R7)

R8 = N[1:-1, 0]
C8 = N[1:-1, 0]
V8 = np.ones_like(R8)

R9 = N[1:-1, -1]
C9 = N[1:-1, -1]
V9 = np.ones_like(R9)

row = np.concatenate((R1,
                      R2, R3, R4, R5,
                      R6, R7, R8, R9))
col = np.concatenate((C1,
                      C2, C3, C4, C5,
                      C6, C7, C8, C9))
val = np.concatenate((V1,
                      V2, V3, V4, V5,
                      V6, V7, V8, V9))

A = sp.csc_matrix((val, (row, col)))

F = f(X, Y)
F[0, :] = phi_L(Y[0, :])
F[-1, :] = phi_R(Y[-1, :])
F[:,  0] = phi_B(X[:, 0])
F[:, -1] = phi_T(X[:, -1])

end_1 = time.time()

beg_2 = time.time()
u = sp.linalg.spsolve(A, F.ravel())
u = u.reshape(X.shape)
end_2 = time.time()

print('fill matrices:  %.5f sec' % (end_1 - beg_1))
print('linalg.spsolve: %.5f sec' % (end_2 - beg_2))


fig = plt.figure(dpi=200, figsize=(8.0, 4.0))

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

options = {'cmap': 'jet',
           'aspect': 'equal',
           'origin': 'lower',
           'extent': [xmin - 0.5 * hx, xmax + 0.5 * hx,
                      ymin - 0.5 * hy, ymax + 0.5 * hy],
           'interpolation': 'nearest'}

ax1.set_title('Решение')
ax1.set_xlim(xmin, xmax)
ax1.set_ylim(ymin, ymax)
img1 = ax1.imshow(u.T, **options)
fig.colorbar(img1, ax=ax1)

ax2.set_title('Погрешность')
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(ymin, ymax)
img2 = ax2.imshow(np.abs(u - u_exact(X, Y)).T, **options)
fig.colorbar(img2, ax=ax2, format='%.2e')

fig.tight_layout()
plt.show()
