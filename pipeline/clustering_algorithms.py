import time
import logging
import numpy as np
import hdbscan
import igraph as ig
import leidenalg as la

from sklearn.cluster import DBSCAN

class HDBSCANClustering:
    def __init__(self, min_cluster_size=5, min_samples=None, metric="euclidean"):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric

    def run(self, data):
        logging.info("Running HDBSCAN clustering...")
        start_time = time.perf_counter()
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            core_dist_n_jobs=-1,
            approx_min_span_tree=True,
        )
        clusterer.fit(data)
        labels = clusterer.labels_
        elapsed_time = time.perf_counter() - start_time
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logging.info(f"HDBSCAN found {num_clusters} clusters in {elapsed_time:.2f} seconds")
        return labels, elapsed_time, num_clusters

class SparseHDBSCANClustering:
    def __init__(self, min_cluster_size=10):
        self.min_cluster_size = min_cluster_size

    def run(self, sparse_matrix):
        logging.info("Running HDBSCAN on sparse k-NN graph...")
        start_time = time.perf_counter()
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric="precomputed",
            core_dist_n_jobs=-1,
            approx_min_span_tree=True,
        )
        clusterer.fit(sparse_matrix)
        labels = clusterer.labels_
        elapsed_time = time.perf_counter() - start_time
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logging.info(f"Sparse HDBSCAN found {num_clusters} clusters in {elapsed_time:.2f} seconds")
        return labels, elapsed_time, num_clusters

class LeidenClustering:
    def __init__(self, resolution=5.0):
        self.resolution = resolution

    def run(self, sparse_matrix):
        logging.info("Running Leiden clustering on sparse k-NN graph...")
        start_time = time.perf_counter()
        num_nodes = sparse_matrix.shape[0]
        # Convert nonzero entries to an edge list
        edges = list(zip(*sparse_matrix.nonzero()))
        graph = ig.Graph(n=num_nodes, edges=edges, directed=False)
        partition = la.find_partition(graph, la.CPMVertexPartition, resolution_parameter=self.resolution)
        labels = np.array(partition.membership)
        elapsed_time = time.perf_counter() - start_time
        num_clusters = len(set(labels))
        logging.info(f"Leiden clustering found {num_clusters} clusters in {elapsed_time:.2f} seconds")
        return labels, elapsed_time, num_clusters



class SparseDBSCANClustering:
    def __init__(self, eps=0.5, min_samples=3):
        """
        eps: Maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples: Number of samples (or total weight) in a neighborhood for a point to be considered 
                     as a core point.
        """
        self.eps = eps
        self.min_samples = min_samples

    def run(self, sparse_matrix):
        """
        Perform DBSCAN clustering on a sparse distance matrix.
        
        Args:
            sparse_matrix (scipy.sparse or array-like): Precomputed distance matrix. 
                                                       Must be shape (n_samples, n_samples).

        Returns:
            labels (ndarray): Cluster labels for each point.
            elapsed_time (float): Time taken for DBSCAN.
            num_clusters (int): Number of clusters (excluding noise).
        """
        logging.info("Running DBSCAN on sparse distance matrix...")
        start_time = time.perf_counter()
        
        # Create the DBSCAN clusterer with a precomputed distance matrix
        dbscan_clusterer = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="precomputed",
            n_jobs=-1  # Use all available CPUs
        )
        
        # Fit the DBSCAN clusterer
        dbscan_clusterer.fit(sparse_matrix)
        
        # Extract labels
        labels = dbscan_clusterer.labels_
        
        # Calculate runtime
        elapsed_time = time.perf_counter() - start_time
        
        # Count number of clusters (label = -1 is noise)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        logging.info(f"Sparse DBSCAN found {num_clusters} clusters in {elapsed_time:.2f} seconds")
        return labels, elapsed_time, num_clusters
