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

# Import feather files
ion_df = pd.read_feather('data/ion.feather')
electron_df = pd.read_feather('data/electron.feather')

# Parse and reshape ions
num_ion = 10000
num_steps = (ion_df.shape[0] // num_ion)

ion_pos = ion_df[['pos_x', 'pos_y', 'pos_z']].values.reshape(num_ion, num_steps, 3)
ion_vel = ion_df[['vel_x', 'vel_y', 'vel_z']].values.reshape(num_ion, num_steps, 3)

print("Ion Positions Shape:", ion_pos.shape)
print("Ion Velocities Shape:", ion_vel.shape)

# Parse and reshape electrons
num_electron = 100
num_steps = (electron_df.shape[0] // num_electron)

electron_pos = electron_df[['pos_x', 'pos_y', 'pos_z']].values.reshape(num_electron, num_steps, 3)
electron_vel = electron_df[['vel_x', 'vel_y', 'vel_z']].values.reshape(num_electron, num_steps, 3)

print("Electron Positions Shape:", electron_pos.shape)
print("Electron Velocities Shape:", electron_vel.shape)

# Prepare ions for ML
ion_x = ion_pos[:, :, 0]
print("X Shape Pre-PCA:", ion_x.shape)
ion_y = ion_pos[:, :, 1]
print("Y Shape Pre-PCA:", ion_y.shape)

'''
# FFT Preprocessing
ppx_x = pos[:, :, 0] / np.max(np.abs(pos[:, :, 0]))  # Normalizes set
ppx_x = fft(ppx_x, axis=1)  # Processes along Y axis
x_mag = np.abs(ppx_x)
ppx_x = (x_mag/np.max(x_mag))  # Normalizes magnitudes since FFT processes with complex numbers.
ppx_x = np.array(x_mag)
print(ppx_x)


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
