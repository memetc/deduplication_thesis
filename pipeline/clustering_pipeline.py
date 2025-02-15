import json
import logging
import numpy as np
from .data_loader import DataLoader
from .umap_reducer import UMAPReducer
from .clustering_algorithms import HDBSCANClustering, SparseHDBSCANClustering, LeidenClustering
from .knn_graph import KNNGraph

class ClusteringPipeline:
    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(column_name=config["data"].get("column_name", "intensities_raw"))
        self.umap_reducer = UMAPReducer(
            n_components=config["umap"]["n_components"],
            n_neighbors=config["umap"]["n_neighbors"],
            min_dist=config["umap"]["min_dist"],
            metric=config["umap"]["metric"],
        )
        self.hdbscan_clusterer = HDBSCANClustering(
            min_cluster_size=config["hdbscan"]["min_cluster_size"],
            min_samples=config["hdbscan"].get("min_samples", None),
            metric=config["hdbscan"]["metric"],
        )
        self.knn_graph = KNNGraph(
            k=config["knn"]["k"],
            use_sqrt=config["knn"]["use_sqrt"],
        )
        self.sparse_hdbscan_clusterer = SparseHDBSCANClustering(
            min_cluster_size=config["sparse_hdbscan"]["min_cluster_size"]
        )
        self.leiden_clusterer = LeidenClustering(
            resolution=config["leiden"]["resolution"]
        )
        self.sparse_dbscan_clusterer = SparseDBSCANClustering(
            eps=config["sparse_dbscan"]["eps"],
            min_samples=config["sparse_dbscan"]["min_samples"]
        )

    def run_umap_hdbscan(self, data):
        logging.info("Starting UMAP reduction for UMAP + HDBSCAN strategy.")
        embedding = self.umap_reducer.reduce(data)
        logging.info("UMAP reduction completed. Embedding shape: %s", np.shape(embedding))
        
        logging.info("Starting HDBSCAN clustering on UMAP embedding.")
        labels, elapsed_time, num_clusters = self.hdbscan_clusterer.run(embedding)
        logging.info("HDBSCAN clustering completed: %d clusters found in %.2f seconds.", num_clusters, elapsed_time)
        
        return {
            "labels": labels.tolist(),
            "time": elapsed_time,
            "num_clusters": num_clusters,
            "embedding": embedding.tolist()
        }
    
    def run_sparse_hdbscan(self, data):
        logging.info("Computing k-NN graph for sparse HDBSCAN strategy.")
        sparse_matrix = self.knn_graph.compute_graph(data)
        logging.info("k-NN graph computed with %d nonzero edges.", sparse_matrix.nnz)
        
        logging.info("Starting sparse HDBSCAN clustering on k-NN graph.")
        labels, elapsed_time, num_clusters = self.sparse_hdbscan_clusterer.run(sparse_matrix)
        logging.info("Sparse HDBSCAN clustering completed: %d clusters found in %.2f seconds.", num_clusters, elapsed_time)
        
        return {"labels": labels.tolist(), "time": elapsed_time, "num_clusters": num_clusters}
    
    def run_leiden(self, data):
        logging.info("Computing k-NN graph for Leiden clustering strategy.")
        sparse_matrix = self.knn_graph.compute_graph(data)
        logging.info("k-NN graph computed with %d nonzero edges.", sparse_matrix.nnz)
        
        logging.info("Starting Leiden clustering on k-NN graph.")
        labels, elapsed_time, num_clusters = self.leiden_clusterer.run(sparse_matrix)
        logging.info("Leiden clustering completed: %d clusters found in %.2f seconds.", num_clusters, elapsed_time)
        
        return {"labels": labels.tolist(), "time": elapsed_time, "num_clusters": num_clusters}

    def run_sparse_dbscan(self, data):
        """
        1. Compute the k-NN graph.
        2. Run SparseDBSCANClustering.
        3. Return a dict with labels, runtime, and number of clusters.
        """
        logging.info("Computing k-NN graph for sparse DBSCAN strategy.")
        sparse_matrix = self.knn_graph.compute_graph(data)
        
        logging.info("k-NN graph computed with %d nonzero edges.", sparse_matrix.nnz)
        
        logging.info("Starting sparse DBSCAN clustering on k-NN graph.")
        labels, elapsed_time, num_clusters = self.sparse_dbscan_clusterer.run(sparse_matrix)
        
        logging.info("Sparse DBSCAN clustering completed: %d clusters found in %.2f seconds.", num_clusters, elapsed_time)
        
        # Return dictionary in the same format as run_sparse_hdbscan
        return {
            "labels": labels.tolist(),   # Convert NumPy array to list
            "time": elapsed_time,
            "num_clusters": num_clusters
        }

    def run_all(self, data):
        results = {}
        results["umap_hdbscan"] = self.run_umap_hdbscan(data)
        results["sparse_hdbscan"] = self.run_sparse_hdbscan(data)
        results["leiden"] = self.run_leiden(data)
        return results

    def save_results(self, results, file_path):
        with open(file_path, "w") as f:
            json.dump({"config": self.config, "results": results}, f, indent=4)
        logging.info(f"Results saved to {file_path}")
