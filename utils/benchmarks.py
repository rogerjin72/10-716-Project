import torch
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from utils.config import *


def preprocess_intermediate(data):
    """
    reshape intermediate layers for clustering
    data = (n, d, x, y)
    
    output shape: (n * x * y, d)
    """
    data = data.reshape(data.shape[:2] + (-1,))
    data = data.transpose(-1, -2).reshape(-1, 64)
    return data.numpy()

 
def get_pca(data, k=10):
    try:
        components = torch.load(DATA_PATH / 'pca.pt')
    except FileNotFoundError:
        pca = PCA(n_components=k)
        pca.fit(data)
        components = torch.Tensor(pca.components_)
        torch.save(components, DATA_PATH / 'pca.pt')
    return components


def get_k_means(data, k=10, max_iter=100000):
    kmeans_dir = DATA_PATH / 'kmeans'
    try:
        kmeans_dir.mkdir()
    except FileExistsError:
        pass

    try:
        centroids = torch.load(kmeans_dir / f'kmeans_{k}.pt')
    except FileNotFoundError:
        kmeans = KMeans(n_clusters=k, n_init=1, max_iter=max_iter, verbose=2)
        kmeans.fit(data)
        centroids = torch.Tensor(kmeans.cluster_centers_)
        torch.save(centroids, kmeans_dir / f'kmeans_{k}.pt')

    return centroids
