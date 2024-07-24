# ml.py - collection of machine learning algorithms for use on data read in from reconnection.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# ML
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.fft import fft
from sklearn.metrics import silhouette_score, silhouette_samples

# Functions


# Function to apply FFT preprocessing to a given array 'arr'
def process(arr):
    ppx = arr / np.max(np.abs(arr), axis=1, keepdims=True)  # Normalizes set
    ppx = fft(ppx, axis=1)  # Processes along Y axis
    mag = np.abs(ppx)
    ppx = mag/np.max(mag, axis=1, keepdims=True)  # Normalizes magnitudes since FFT processes with complex numbers.
    ppx = np.array(ppx) # Makes sure it is a numpy array (sometimes likes to not be)
    return ppx


# Function to fit a given array to PCA with a given number of components.
def pca_(arr, com):
    # Performing PCA
    pca = PCA(n_components=com)
    pca__ = pca.fit_transform(arr)
    return pca__


# Function to visualize PC I, II, and variance
def visualize_pca(arr):
    pca = PCA()
    score = pca.fit_transform(arr)
    cumsum = np.cumsum(pca.explained_variance_ratio_)

    # Plot PCA results
    plt.figure()
    plt.scatter(score[:, 0], score[:, 1], marker='.')
    plt.xlabel('PC I')
    plt.ylabel('PC II')
    plt.grid(True)
    plt.title('PCA of FFT-Processed Array')
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
    print("Component Number needed for 98.5% variance:", d)


# Function to cluster array
def kmeans_cluster(nc, arr, seed):
    kmeans = KMeans(n_clusters=nc, n_init=20, random_state=seed)
    labels = kmeans.fit_predict(arr)
    return labels


# Function to visualize clusters
def visualize_kmeans(nc, arr, col, seed, plots, num_plot):
    kmeans = KMeans(n_clusters=nc, n_init=20, random_state=seed)
    labels = kmeans.fit_predict(arr)

    ncols = col
    nrows = (kmeans.n_clusters + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, sharex=True,
                             sharey=True)  # Creates a grid of subplots based on column and row
    axes = axes.flatten()

    for cluster in range(kmeans.n_clusters):
        ax = axes[cluster]
        for i, plot in enumerate(plots):
            if labels[i] == cluster:
                ax.plot(plot[:, 1], plot[:, 0])
        ax.set_title('Cluster {:} - {:2.2%}'.format(cluster + 1, (labels == cluster).sum() / num_plot))
        ax.set_aspect('equal')
        ax.set_xlim(150, 350)
        ax.set_ylim(90, 110)
        #ax.set_aspect(1)

    for j in range(kmeans.n_clusters, len(axes)): fig.delaxes(axes[j])  # Removes empty subplots from the figure
    return fig


# Function to determine and visualize kmeans clusters and do silhouette analysis on the clusters
def visualize_sil(nc, arr, col, seed, plots, num_plot, num_sil):
    kmeans = KMeans(n_clusters=nc, n_init=20, random_state=seed)
    labels = kmeans.fit_predict(arr)

    silhouette_avg = silhouette_score(arr, labels)
    print("Silhouette Score: {}".format(silhouette_avg))  # Average score for how closely a sample fits its cluster.
    silh_v = silhouette_samples(arr, labels)

    # Grabs the best fit graphs per cluster
    best_samples = np.zeros((num_sil, kmeans.n_clusters))
    for i in range(kmeans.n_clusters):
        clus_silh_val = silh_v[labels == i]  # Grabs all silhouette values for all graphs in cluster
        cluster_i = np.where(labels == i)[0]  # Grabs the index of the cluster
        sorted_i = cluster_i[np.argsort(-clus_silh_val)]  # Descending order
        best_samples[:, i] = sorted_i[:num_sil]  # Grabs the first 'num_sil' samples and throws it into the row

    # Plotting clusters
    ncols = col
    nrows = (kmeans.n_clusters + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8), sharex=True,
                             sharey=True)  # Creates a grid of subplots based on column and row
    axes = axes.flatten()

    for cluster in range(kmeans.n_clusters):
        ax = axes[cluster]
        for s in best_samples[:, cluster]:
            ax.plot(plots[int(s), :, 1], plots[int(s), :, 0])
        ax.set_title('Cluster {:} - {:2.2%}'.format(cluster + 1, (labels == cluster).sum() / num_plot))

    for j in range(kmeans.n_clusters, len(axes)): fig.delaxes(axes[j])  # Removes empty subplots from the figure

    return fig


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

# Plot to check if parsed correctly
plot_bool = False
if plot_bool:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 200)
    ax.set_aspect(1)
    for pos in ion_pos:
        ax.plot(pos[:, 1], pos[:, 0])
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 200)
    ax.set_aspect(1)
    for pos in electron_pos:
        ax.plot(pos[:, 1], pos[:, 0])
    plt.tight_layout()
    plt.show()

'''
# Prepare ions for ML
ion_x = ion_pos[:, :, 1]
ion_y = ion_pos[:, :, 0]

ion_vel_x = ion_vel[:, :, 1]
ion_vel_y = ion_vel[:, :, 0]

ppx_x = process(ion_x)
ppx_y = process(ion_y)
ppx_vx = process(ion_vel_x)
ppx_vy = process(ion_vel_y)
# Combine Results
combined = np.hstack((ppx_x, ppx_y, ppx_vx, ppx_vy))
print("Combined Shape Pre-PCA:", combined.shape)

# PCA
score = pca_(combined, 10)
score = normalize(score)
print("Combined Shape Post-PCA:", score.shape)

# K-Means

fig = visualize_kmeans(12, score, 4, seed=42)
plt.suptitle("Clustering based on Both Position and Velocity. Display based on Silhouette.")
plt.tight_layout()
plt.show()

# Silhouette Analysis

fig = visualize_sil(12, score, 4, 42, ion_pos, num_ion, 50)
plt.suptitle("Clustering based on Both Position and Velocity. Display based on Silhouette.")
plt.tight_layout()
plt.show()
'''

# Prepare electrons for ML
electron_x = electron_pos[:, :, 1]
electron_y = electron_pos[:, :, 0]

electron_vel_x = electron_vel[:, :, 1]
electron_vel_y = electron_vel[:, :, 0]

# FFT
ppx_x = process(electron_x)
ppx_y = process(electron_y)
ppx_vx = process(electron_vel_x)
ppx_vy = process(electron_vel_y)

# Combining
comb1 = np.hstack((ppx_x, ppx_y))
comb2 = np.hstack((ppx_vx, ppx_vy))
comb3 = np.hstack((ppx_x, ppx_y, ppx_vx, ppx_vy))

#visualize_pca(comb1)
#visualize_pca(comb2)
#visualize_pca(comb3)

# PCA
score1 = pca_(comb1, 14)
score1 = normalize(score1)
score2 = pca_(comb2, 74)
score2 = normalize(score2)
score3 = pca_(comb3, 74)
score3 = normalize(score3)

#Clustering
fig1 = visualize_kmeans(6, score1, 3, 42, electron_pos, num_electron)
plt.suptitle("Electron Clustering Based on Position")
plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.show()

fig2 = visualize_kmeans(6, score2, 3, 42, electron_pos, num_electron)
plt.suptitle("Electron Clustering Based on Velocity")
plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.show()

fig3 = visualize_kmeans(6, score3, 3, 42, electron_pos, num_electron)
plt.suptitle("Electron Clustering Based on Both Position and Velocity")
plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.show()
