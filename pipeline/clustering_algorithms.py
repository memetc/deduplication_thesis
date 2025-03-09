import time
import logging
import numpy as np
import hdbscan
import igraph as ig
import leidenalg as la

from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch


class BIRCHClustering:
    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=None, compute_labels=False):
        """
        :param threshold: The radius threshold that defines whether a
                          new subcluster should be started.
        :param branching_factor: Maximum number of subclusters in each node.
        :param n_clusters: Number of final clusters. If set to None, the
                           subclusters are not further aggregated.
        """
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.compute_labels = compute_labels

    def run(self, data, chunk_size=100000):
        logging.info("Running BIRCH clustering (partial-fit mode)...")
        start_time = time.perf_counter()

        # Initialize BIRCH with compute_labels=False so partial_fit doesn't try
        # to maintain labels across iterations. We'll predict at the end.
        clusterer = Birch(
            threshold=self.threshold,
            branching_factor=self.branching_factor,
            n_clusters=self.n_clusters,
            compute_labels=self.compute_labels
        )

        # Step 1: Partial-fit the BIRCH model in batches
        n_samples = data.shape[0]
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = start_idx + chunk_size
            chunk = data[start_idx:end_idx]
            clusterer.partial_fit(chunk)
        
        # Step 2: Once the CF tree is built, predict final labels for all data
        labels = clusterer.predict(data)

        elapsed_time = time.perf_counter() - start_time
        num_clusters = len(set(labels))

        logging.info(f"BIRCH found {num_clusters} clusters in {elapsed_time:.2f} seconds")
        return labels, elapsed_time, num_clusters

class HDBSCANClustering:
    def __init__(self, min_cluster_size=3,cluster_selection_epsilon=0.1, min_samples=None, metric="euclidean", cluster_selection_method="leaf"):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method
        
    def run(self, data):
        logging.info("Running HDBSCAN clustering...")
        start_time = time.perf_counter()
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            core_dist_n_jobs=-1,
            approx_min_span_tree=True,
            cluster_selection_method=self.cluster_selection_method
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


class DBSCANClustering:
    def __init__(self, eps=0.5, min_samples=5, metric="euclidean"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def run(self, data):
        logging.info("Running DBSCAN clustering...")
        start_time = time.perf_counter()

        # Create the DBSCAN clusterer
        clusterer = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            n_jobs=-1,  # Parallelize distance computations if desired
        )

        # Fit to data
        clusterer.fit(data)
        labels = clusterer.labels_

        elapsed_time = time.perf_counter() - start_time
        # Count the number of clusters (excluding noise, labeled as -1)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        logging.info(f"DBSCAN found {num_clusters} clusters in {elapsed_time:.2f} seconds")

        return labels, elapsed_time, num_clusters
