import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from numba import jit, njit, prange

# ML
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.fft import fft
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE

str_time = time.time()

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

print(f"Files read and parsed in {time.time() - str_time} seconds.")
print("-------------")

'''
# Plot to check if parsed correctly
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
ax.set_xlim(0, 512)
ax.set_ylim(0, 200)
ax.set_aspect(1)
for pos in ion_pos:
    ax.plot(pos[:, 1], pos[:, 0])
plt.tight_layout()
plt.show()
'''

# Prepare ions for ML
ion_x = ion_pos[:, :, 1]
ion_y = ion_pos[:, :, 0]

ion_vel_x = ion_vel[:, :, 1]
ion_vel_y = ion_vel[:, :, 0]

# FFT Preprocessing

def process(arr):
    ppx = arr / np.max(np.abs(arr), axis=1, keepdims=True)  # Normalizes set
    ppx = fft(ppx, axis=1)  # Processes along Y axis
    mag = np.abs(ppx)
    ppx = mag/np.max(mag, axis=1, keepdims=True)  # Normalizes magnitudes since FFT processes with complex numbers.
    ppx = np.array(ppx) # Makes sure it is a numpy array (sometimes likes to not be)
    return ppx


# X
ppx_x = process(ion_x)
ppx_y = process(ion_y)
ppx_vx = process(ion_vel_x)
ppx_vy = process(ion_vel_y)
# Combine Results
combined = np.hstack((ppx_x, ppx_y, ppx_vx, ppx_vy))
print("Combined Shape Pre-PCA:", combined.shape)

# PCA

# Performing PCA
pca = PCA(n_components=10)
score = pca.fit_transform(combined)
score = normalize(score)
print("Combined Shape Post-PCA:", score.shape)

'''
cumsum = np.cumsum(pca.explained_variance_ratio_)

# Plot PCA results
plt.figure()
plt.scatter(score[:, 0], score[:, 1], marker='.')
plt.xlabel('PC I')
plt.ylabel('PC II')
plt.grid(True)
plt.title('PCA of FFT-Processed Ion Positions')
plt.show()


# Plot Explained Variance
xmax = np.argmax(cumsum >= 0.999) + 1
plt.figure()
plt.plot(cumsum, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.grid(True)
plt.xlim(0, xmax)
plt.show()

d = np.argmax(cumsum >= 0.985) + 1
print(d)
'''

# K-Means

# Clustering
kmeans = KMeans(n_clusters=12, n_init=20, random_state=42)
labels = kmeans.fit_predict(score)

# Plotting clusters
ncols = 4
nrows = (kmeans.n_clusters + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True) # Creates a grid of subplots based on column and row
axes = axes.flatten()

for cluster in range(kmeans.n_clusters):
    ax = axes[cluster]
    for i, ion in enumerate(ion_pos):
        if labels[i] == cluster:
            ax.plot(ion[:, 1], ion[:, 0])
    ax.set_title('Cluster {:} - {:2.2%}'.format(cluster + 1, (labels == cluster).sum() / num_ion))
    ax.set_aspect('equal')
    ax.set_xlim(50, 450)
    ax.set_ylim(0,200)
    ax.set_aspect(1)

for j in range(kmeans.n_clusters, len(axes)): fig.delaxes(axes[j])  # Removes empty subplots from the figure
fig.tight_layout()
plt.show()


# Silhouette Analysis

silhouette_avg = silhouette_score(score, labels)
print("Silhouette Score: {}".format(silhouette_avg))  # Average score for how closely a sample fits its cluster. It get's pretty low, but we look at the larger examples anyway.
silh_v = silhouette_samples(score, labels)


# Grabs the best fit graphs per cluster
best_samples = np.zeros((50, kmeans.n_clusters))  # 5 x cluster 2D matrix
for i in range(kmeans.n_clusters):
    clus_silh_val = silh_v[labels == i]  # Grabs all silhouette values for all graphs in cluster
    cluster_i = np.where(labels == i)[0]  # Grabs the index of the cluster
    sorted_i = cluster_i[np.argsort(-clus_silh_val)]  # Descending order
    best_samples[:, i] = sorted_i[:50]  # Grabs the first 25 and throws it into the row

# Plotting clusters
ncols = 4
nrows = (kmeans.n_clusters + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8), sharex=True, sharey=True) # Creates a grid of subplots based on column and row
axes = axes.flatten()

for cluster in range(kmeans.n_clusters):
    ax = axes[cluster]
    for s in best_samples[:, cluster]:
        ax.plot(ion_pos[int(s), :, 1], ion_pos[int(s), :, 0])
    ax.set_title('Cluster {:} - {:2.2%}'.format(cluster + 1, (labels == cluster).sum() / num_ion))

for j in range(kmeans.n_clusters, len(axes)): fig.delaxes(axes[j])  # Removes empty subplots from the figure

plt.suptitle("Clustering based on Both Position and Velocity. Display based on Silhouette.")
plt.tight_layout()
plt.show()
