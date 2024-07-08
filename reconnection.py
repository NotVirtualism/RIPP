import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from numba import jit, njit, prange
from scipy.interpolate import RectBivariateSpline


nproc = 8
nx = 200
ny = 64
dx = 0.4
dy = dx
lx = dx * nx
ly = dy * ny
it = 4

### EM fields and density ###
for irank in range(nproc):
    f1 = open('data/bx%05drank=%04d.d' % (it, irank), 'rb')
    f2 = open('data/by%05drank=%04d.d' % (it, irank), 'rb')
    f3 = open('data/bz%05drank=%04d.d' % (it, irank), 'rb')
    f4 = open('data/ex%05drank=%04d.d' % (it, irank), 'rb')
    f5 = open('data/ey%05drank=%04d.d' % (it, irank), 'rb')
    f6 = open('data/ez%05drank=%04d.d' % (it, irank), 'rb')

    dat1 = np.fromfile(f1, dtype='float64').reshape(-1, nx)
    dat2 = np.fromfile(f2, dtype='float64').reshape(-1, nx)
    dat3 = np.fromfile(f3, dtype='float64').reshape(-1, nx)
    dat4 = np.fromfile(f4, dtype='float64').reshape(-1, nx)
    dat5 = np.fromfile(f5, dtype='float64').reshape(-1, nx)
    dat6 = np.fromfile(f6, dtype='float64').reshape(-1, nx)

    if (irank == 0):
        bx = np.copy(dat1)
        by = np.copy(dat2)
        bz = np.copy(dat3)
        ex = np.copy(dat4)
        ey = np.copy(dat5)
        ez = np.copy(dat6)
    else:
        bx = np.concatenate((bx, dat1), axis=0)
        by = np.concatenate((by, dat2), axis=0)
        bz = np.concatenate((bz, dat3), axis=0)
        ex = np.concatenate((ex, dat4), axis=0)
        ey = np.concatenate((ey, dat5), axis=0)
        ez = np.concatenate((ez, dat6), axis=0)

plot_bool = False
if plot_bool:
    # Plotting Magnetic Fields
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=3)
    im1 = ax[0].pcolormesh(bx)
    im2 = ax[1].pcolormesh(by)
    im3 = ax[2].pcolormesh(bz)

    fig.colorbar(im1, ax=ax[0])
    fig.colorbar(im2, ax=ax[1])
    fig.colorbar(im3, ax=ax[2])

    ax[0].set_title('Bx')
    ax[1].set_title('By')
    ax[2].set_title('Bz')
    fig.suptitle('Magnetic Fields')
    plt.show()

    # Plotting Electric Fields
    fig, ax = plt.subplots(figsize=(10, 8), nrows=1, ncols=3)
    im1 = ax[0].pcolormesh(ex)
    im2 = ax[1].pcolormesh(ey)
    im3 = ax[2].pcolormesh(ez)

    fig.colorbar(im1, ax=ax[0])
    fig.colorbar(im2, ax=ax[1])
    fig.colorbar(im3, ax=ax[2])

    ax[0].set_title('Ex')
    ax[1].set_title('Ey')
    ax[1].set_title('Ey')
    ax[2].set_title('Ez')
    fig.suptitle('Electric Fields')
    plt.show()

nop = 100   # number of particles
qi = 1.0    # ion charge
mi = 1.0    # ion mass
qe = -1.0   # electron charge
me = 0.01   # electron mass
c = 1.0     # speed of light in a vacuum (1.0 for normalized)
dt = 0.01   # time step
nt = 10000  # number of time steps

# Particle attributes are stored as arrays of arrays
e_pos = np.zeros((nop, nt, 3))
e_vel = np.zeros((nop, nt, 3))
i_pos = np.zeros((nop, nt, 3))
i_vel = np.zeros((nop, nt, 3))


@njit(parallel=True)
def simulate_particles(pos_s, vel_s, nop, nt, m, q, dt, c):
    for p in prange(nop):
        x = np.array([100.0, 256.0, 256.0])  # initial particle position
        v = np.random.randn(3)  # initial particle velocity

        for step in range(nt):
            # Store current position and velocity
            pos_s[p, step] = x
            vel_s[p, step] = v

            # Interpolate fields at particle position using nearest neighbor
            check = np.array([round(i) for i in x])

            if(check[0] > 200 or check[0] < 0 or check[1] > 512 or check[1] < 0): # Break if outside bounds
                break
            E = np.array([ex[check[1], check[0]], ey[check[1], check[0]], ez[check[1], check[0]]])
            B = np.array([bx[check[1], check[0]], by[check[1], check[0]], bz[check[1], check[0]]])

            # Half-step for velocity.
            v_minus = v + (q / m) * E * (dt / 2)

            # Accounting for magnetic field.
            T = (dt / 2) * (q * B / (m * c))
            S = 2 * T / (1 + np.dot(T, T))
            v_prime = v_minus + np.cross(v_minus, T)
            v_plus = v_minus + np.cross(v_prime, S)

            # Full-step for velocity.
            v = v_plus + (q * E / m) * (dt / 2)

            # Position change
            x = x + v * dt


simulate_particles(i_pos, i_vel, nop, nt, mi, qi, dt, c)
# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
ax.set_xlim(0, 512)
ax.set_ylim(0, 200)
for pos in i_pos:
    ax.plot(pos[:, 1], pos[:, 0])

plt.tight_layout()
plt.show()
