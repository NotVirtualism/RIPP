import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from numba import jit, njit, prange

# ML
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.fft import fft
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE

# Variables for EM fields
nproc = 8
nx = 200
ny = 64
dx = 0.4
dy = dx
lx = dx * nx
ly = dy * ny
it = 4

# Importing EM Fields
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

nop = 10000  # number of particles
qi = 1.0     # ion charge
mi = 1.0     # ion mass
qe = -1.0    # electron charge
me = 0.01    # electron mass
c = 1.0      # speed of light in a vacuum (1.0 for normalized)
dt = 0.01    # time step
nt = 10000   # number of time steps

# Particle attributes are stored as arrays of arrays
e_pos = np.zeros((nop//100, nt*100, 3))
e_vel = np.zeros((nop//100, nt*100, 3))
i_pos = np.zeros((nop, nt, 3))
i_vel = np.zeros((nop, nt, 3))


@njit(parallel=True)
def simulate_particles(pos_s, vel_s, nop, nt, m, q, dt, c):
    for p in prange(nop):
        x = np.array([100.0 + np.random.randint(-10, 11), 256.0 + np.random.randint(-10, 11), 256.0])  # initial particle position (given a 10 unit random variance)
        v = np.random.randn(3)  # initial particle velocity
        run = True

        for step in range(nt):
            # Store current position and velocity
            pos_s[p, step] = x
            vel_s[p, step] = v

            # Interpolate fields at particle position using nearest neighbor
            check = np.array([round(i) for i in x])

            if(check[0] >= nx or check[0] < 0 or check[1] >= ny*nproc or check[1] < 0): # Break if outside bounds
                run = False
            if run:
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


start_time = time.time()
simulate_particles(i_pos, i_vel, nop, nt, mi, qi, dt, c)
print("--- Ions simulated in %s seconds ---" % (time.time() - start_time))
start_time = time.time()
simulate_particles(e_pos, e_vel, nop//100, nt*100, me, qe, dt/100, c)
print("--- Electrons simulated in %s seconds ---" % (time.time() - start_time))

# Plotting ion and electron paths on XY axis slice.
plot2_bool = False
if plot2_bool:
    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 200)
    ax.set_aspect(1)
    for pos in i_pos:
        ax.plot(pos[2000:5001:10, 1], pos[2000:5001:10, 0]) # 300 timesteps

    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 200)
    for pos in e_pos:
        ax.plot(pos[200000:500001:100, 1], pos[200000:500001:100, 0]) # 3000 timesteps

    plt.tight_layout()
    plt.show()

# ML Algorithms (for implementation later)
'''
def ml(pos, vel):
    # FFT Preprocessing
    ppx = vel / np.max(np.abs(vel))  # Normalizes set
    ppx = fft(ppx, axis=1)  # Processes along Y axis
    fft_magnitude = np.abs(ppx)
    ppx = (fft_magnitude/np.max(fft_magnitude))  # Normalizes magnitudes since FFT processes with complex numbers.
    ppx = np.array(ppx)
    
    
    # PCA
    pca = PCA(n_components=40)  # 40 components keeps 98.5% of variance.
    pca_r = pca.fit_transform(ppx)
    
    print("Shape Post-PCA: {}".format(np.shape(pca_r)))
    
    # K-Means
    kmeans = KMeans(n_clusters=3, n_init=10)
    labels = kmeans.fit_predict(pca_r)
    
    # Silhouette Analysis
    silhouette_avg = silhouette_score(pca_r, labels)
    print("Silhouette Score: {}".format(silhouette_avg))  # Average score for how closely a sample fits its cluster. It get's pretty low, but we look at the larger examples anyway.
    silh_v = silhouette_samples(pca_r, labels)
    
    
    # Grabs the best fit graphs per cluster
    best_samples = np.zeros((5, kmeans.n_clusters))  # 5 x cluster 2D matrix
    for i in range(kmeans.n_clusters):
        clus_silh_val = silh_v[labels == i]  # Grabs all silhouette values for all graphs in cluster
        cluster_i = np.where(labels == i)[0]  # Grabs the index of the cluster
        sorted_i = cluster_i[np.argsort(-clus_silh_val)]  # Descending order
        best_samples[:, i] = sorted_i[:5]  # Grabs the first 5 and throws it into the row
    
    # Plotting clusters
    ncols = 4
    nrows = (kmeans.n_clusters + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8), sharex=True, sharey=False) # Creates a grid of subplots based on column and row
    axes = axes.flatten()
    
    for cluster in range(kmeans.n_clusters):
        ax = axes[cluster]
        for s in best_samples[:, cluster]:
            ax.plot(pos[int(s), :])
        ax.set_title('Cluster {:} - {:2.2%}'.format(cluster + 1, (labels == cluster).sum() / (nop*3)))
    
    for j in range(kmeans.n_clusters, len(axes)): fig.delaxes(axes[j])  # Removes empty subplots from the figure
    
    plt.tight_layout()
    plt.show()
'''
