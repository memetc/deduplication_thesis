import json
import logging
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

    def run_umap_hdbscan(self, data):
        embedding = self.umap_reducer.reduce(data)
        labels, elapsed_time, num_clusters = self.hdbscan_clusterer.run(embedding)
        return {
            "labels": labels.tolist(),
            "time": elapsed_time,
            "num_clusters": num_clusters,
            "embedding": embedding.tolist()
        }

    def run_sparse_hdbscan(self, data):
        sparse_matrix = self.knn_graph.compute_graph(data)
        labels, elapsed_time, num_clusters = self.sparse_hdbscan_clusterer.run(sparse_matrix)
        return {"labels": labels.tolist(), "time": elapsed_time, "num_clusters": num_clusters}

    def run_leiden(self, data):
        sparse_matrix = self.knn_graph.compute_graph(data)
        labels, elapsed_time, num_clusters = self.leiden_clusterer.run(sparse_matrix)
        return {"labels": labels.tolist(), "time": elapsed_time, "num_clusters": num_clusters}

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
