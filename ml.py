"""
this is a template. I am currently just moving respective code chunks to their own files for making work later.
"""

# Machine learning
"""
Apply machine learning algorithms to segments of the timesteps and particles to group them
"""
if ml_bool:
    # Segmenting the full algorithm to timestep chunks
    tsc = 100  # Time Step Chunk: Segments will be grouped by x tsc (100 position points per segment by default).
    # MAKE SURE NT % tsc = 0
    segments = np.reshape(pos, (nt // tsc, tsc, 3))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    for s in segments:
        ax.plot(s[:, 0], s[:, 1])
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.show()

    # Apply FFT to each segment and extract features
    fft_segments = []
    for segment in segments:
        fft_result = fft(segment, axis=0)
        fft_magnitude = np.abs(fft_result)
        fft_segments.append(fft_magnitude.flatten())

    fft_segments = np.array(fft_segments)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=2)  # Adjust the number of components as needed
    pca_result = pca.fit_transform(fft_segments)

    """
    To help find the best number of components, use:
    pca = PCA()
    pca.fit(x_train)
    cumsum = np.cumsum(pca.explained_variance_ratio_) - It is also a good idea to plot the results of this.
    d = np.argmax(cumsum >= 0.95) + 1

    The optimal number of components is reaches when the cumulative variance stops growing fast.
    Change the 0.95 to whatever percentage of variance you want (99% - 0.99, 98% - 0.98, etc.)
    """

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=2)  # Adjust the number of clusters as needed
    kmeans_result = kmeans.fit_predict(pca_result)

    # Plot the clustered segments
    num_clusters = len(np.unique(kmeans_result))
    ncols = 3
    nrows = (num_clusters + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'orange', 'purple', 'pink']
    for cluster in range(num_clusters):
        ax = axes[cluster]
        for i, segment in enumerate(segments):
            if kmeans_result[i] == cluster:
                ax.plot(segment[:, 0], segment[:, 1], color=colors[cluster])
        ax.set_title(f'Cluster {cluster + 1}')
        ax.set_box_aspect(1)

    for j in range(num_clusters, len(axes)): fig.delaxes(axes[j])  # Removes empty subplots from the figure

    plt.tight_layout()
    plt.show()

