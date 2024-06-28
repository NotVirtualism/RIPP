# Purpose: Use unsupervised machine learning techniques to group and classify Scaled Brownian Motion

from scipy.special import erfcinv
import numpy as np
import matplotlib.pyplot as plt

# ML
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.fft import fft
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.manifold import TSNE


def SBM(nop, nt, a):
    dp = np.arange(nt) + 1
    dm = np.arange(nt)

    xx = np.zeros((nop, nt))
    for ip in range(nop):
        dx = dp**a-dm**a
        dx = np.sqrt(2*dx)*erfcinv(2-2*np.random.rand(nt))
        xx[ip, :] = np.cumsum(dx)
        xx[ip, :] = xx[ip, :]-xx[ip, 0]  # shift particle trajectory so initial position is at origin
    return xx

def MSD(x):
    msd = np.sum(x ** 2, axis=0) / len(x)

    return msd


nop = 1000
nt = 4096 + 1
nor = SBM(nop,nt,a=1.0)
sub = SBM(nop,nt,a=0.5)
sup = SBM(nop,nt,a=1.5)
all_b = np.concatenate((nor,sub,sup))
print("Shape Pre-PCA: {}".format(np.shape(all_b)))


# FFT Preprocessing
ppx = all_b / np.max(np.abs(all_b))  # Normalizes set
ppx = fft(ppx, axis=1)
fft_magnitude = np.abs(ppx)
ppx = (fft_magnitude/np.max(fft_magnitude))
ppx = np.array(ppx)


# PCA
pca = PCA(n_components=40)
pca_r = pca.fit_transform(ppx)

print("Shape Post-PCA: {}".format(np.shape(pca_r)))

# K-Means
kmeans = KMeans(n_clusters=12, n_init=10)
labels = kmeans.fit_predict(pca_r)

# Silhouette Analysis
silhouette_avg = silhouette_score(pca_r, labels)
print("Silhouette Score: {}".format(silhouette_avg))
silh_v = silhouette_samples(pca_r, labels)


# Grabs the best fit graphs per cluster
best_samples = np.zeros((5, kmeans.n_clusters)) # 10 x cluster 2D matrix
for i in range(kmeans.n_clusters):
    clus_silh_val = silh_v[labels == i]  # Grabs all silhouette values for all graphs in cluster
    cluster_i = np.where(labels == i)[0]  # Grabs the index of the cluster
    sorted_i = cluster_i[np.argsort(-clus_silh_val)]  # Descending order
    best_samples[:, i] = sorted_i[:5]  # Grabs the first 5 and throws it into the row

ncols = 4
nrows = (kmeans.n_clusters + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8), sharex=True, sharey=False)
axes = axes.flatten()

for cluster in range(kmeans.n_clusters):
    ax = axes[cluster]
    for s in best_samples[:, cluster]:
        ax.plot(all_b[int(s), :])
    ax.set_title('Cluster {:} - {:2.2%}'.format(cluster + 1, (labels == cluster).sum() / (nop*3)))
    #ax.set_box_aspect()

for j in range(kmeans.n_clusters, len(axes)): fig.delaxes(axes[j])  # Removes empty subplots from the figure

plt.tight_layout()
plt.show()


'''
Visualizing Clusters
# t-SNE for Visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_r = tsne.fit_transform(pca_r)

# Plotting
plt.figure(figsize=(10, 7))
plt.scatter(tsne_r[:, 0], tsne_r[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
plt.title('Cluster visualization using t-SNE')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(label='Cluster Label')
plt.show()
'''

'''
#Scree Plot
PC_vals = np.arange(pca.n_components_) + 1
plt.plot(PC_vals, pca.explained_variance_ratio_, 'o-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()
'''
