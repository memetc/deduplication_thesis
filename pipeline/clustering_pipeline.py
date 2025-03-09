import json
import logging
import numpy as np
import time
import hdbscan

from .data_loader import DataLoader
from .umap_reducer import UMAPReducer
from .clustering_algorithms import HDBSCANClustering, SparseHDBSCANClustering, DBSCANClustering, LeidenClustering, BIRCHClustering
from .knn_graph import KNNGraph
from .utils import save_embedding
from sklearn.cluster import DBSCAN
from typing import Optional

class ClusteringPipeline:
    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(column_name=config["data"].get("column_name", "intensities_raw"))
        self.umap_reducer = UMAPReducer(
            n_components=config["umap"]["n_components"],
            n_neighbors=config["umap"]["n_neighbors"],
            min_dist=config["umap"]["min_dist"],
            metric=config["umap"]["metric"],
            embedding_file = config["umap"]["embedding_file"]
        )
        self.hdbscan_clusterer = HDBSCANClustering(
            min_cluster_size=config["hdbscan"]["min_cluster_size"],
            min_samples=config["hdbscan"].get("min_samples", None),
            metric=config["hdbscan"]["metric"],
            cluster_selection_epsilon=config["hdbscan"]["cluster_selection_epsilon"],
            cluster_selection_method=config["hdbscan"]["cluster_selection_method"]
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
        self.dbscan_clusterer = DBSCANClustering(
            eps=config["sparse_dbscan"]["eps"],
            min_samples=config["sparse_dbscan"]["min_samples"],
            metric="euclidean"
        )
        self.birch_clusterer = BIRCHClustering(
            threshold=config["birch"]["threshold"],       # Very small threshold for granular subclusters   
            branching_factor=config["birch"]["branching_factor"],  # Default
            n_clusters=None,      # Keep raw subclusters, no final merging step
            compute_labels=False
        )

    def run_umap_hdbscan(self, data):
        logging.info("Starting UMAP reduction for UMAP + HDBSCAN strategy.")
        embedding = self.umap_reducer.reduce(data)
        embedding_path = save_embedding(embedding, self.config)

        umap_file = f"umap_model_{int(time.time())}.joblib"
        self.umap_reducer.save(umap_file)

        logging.info("UMAP reduction completed. Embedding shape: %s", np.shape(embedding))
        
        logging.info("Starting HDBSCAN clustering on UMAP embedding.")
        labels, elapsed_time, num_clusters = self.hdbscan_clusterer.run(embedding)
        logging.info("HDBSCAN clustering completed: %d clusters found in %.2f seconds.", num_clusters, elapsed_time)
        
        return {
            "labels": labels.tolist(),
            "time": elapsed_time,
            "num_clusters": num_clusters,
            "embedding": embedding.tolist(),
            "umap_model_file": umap_file

        }
    
    def run_sparse_hdbscan(self, data):
        logging.info("Computing k-NN graph for sparse HDBSCAN strategy.")
        sparse_matrix = self.knn_graph.compute_graph(data)
        logging.info("k-NN graph computed with %d nonzero edges.", sparse_matrix.nnz)
        
        logging.info("Starting sparse HDBSCAN clustering on k-NN graph.")
        labels, elapsed_time, num_clusters = self.dbscan_clusterer.run(sparse_matrix)
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
        labels, elapsed_time, num_clusters = self.dbscan_clusterer.run(sparse_matrix)
        
        logging.info("Sparse DBSCAN clustering completed: %d clusters found in %.2f seconds.", num_clusters, elapsed_time)
        
        # Return dictionary in the same format as run_sparse_hdbscan
        return {
            "labels": labels.tolist(),   # Convert NumPy array to list
            "time": elapsed_time,
            "num_clusters": num_clusters
        }

    def run_umap_dbscan(self, data):
        """
        1. Apply UMAP dimensionality reduction.
        2. Run DBSCAN on the resulting embedding.
        3. Return a dict with labels, runtime, number of clusters, and embedding.
        """
        logging.info("Starting UMAP reduction for UMAP + DBSCAN strategy.")
        embedding = self.umap_reducer.reduce(data)
        embedding_path = save_embedding(embedding, self.config)
        umap_file = f"umap_model_{int(time.time())}.joblib"
        self.umap_reducer.save(umap_file)
        
        #self.umap_reducer.plot_embedding(save_path=f"{embedding_path}_plot.png")
        logging.info("UMAP reduction completed, saved and plotted. Embedding shape: %s", np.shape(embedding))

        logging.info("Starting DBSCAN clustering on UMAP embedding.")
        labels, elapsed_time, num_clusters = self.dbscan_clusterer.run(embedding)
        logging.info("DBSCAN clustering completed: %d clusters found in %.2f seconds.", num_clusters, elapsed_time)

        return {
            "labels": labels.tolist(),  # Convert NumPy array to list
            "time": elapsed_time,
            "num_clusters": num_clusters,
            "embedding": embedding.tolist(),  # Optional: include the UMAP embedding
            "umap_model_file": umap_file
        }


    def run_umap_birch(self, data, chunk_size=100000):
        """
        1. Apply UMAP dimensionality reduction.
        2. Run BIRCH on the resulting embedding.
        3. Return a dict with labels, runtime, number of clusters, and embedding.
        """
        logging.info("Starting UMAP reduction for UMAP + BIRCH strategy.")
        embedding = self.umap_reducer.reduce(data)
        embedding_path = save_embedding(embedding, self.config)
        umap_file = f"umap_model_{int(time.time())}.joblib"
        self.umap_reducer.save(umap_file)
        
        # Optional: Generate a plot of the embedding
        # self.umap_reducer.plot_embedding(save_path=f"{embedding_path}_plot.png")
        logging.info("UMAP reduction completed, saved and plotted. Embedding shape: %s", np.shape(embedding))

        logging.info("Starting BIRCH clustering on UMAP embedding.")
        labels, elapsed_time, num_clusters = self.birch_clusterer.run(embedding, chunk_size)
        logging.info("BIRCH clustering completed: %d clusters found in %.2f seconds.", num_clusters, elapsed_time)

        return {
            "labels": labels.tolist(),   # Convert NumPy array to Python list
            "time": elapsed_time,
            "num_clusters": num_clusters,
            "embedding": embedding.tolist(),  # Optional: include the UMAP embedding
            "umap_model_file": umap_file
        }


    def run_umap_dbscan_with_subclustering(self,
                                           data, 
                                           min_desired_clusters=1000000,
                                           subclustering_eps=0.05,
                                           subclustering_min_samples=3):
        """
        1. Apply UMAP dimensionality reduction.
        2. Run DBSCAN on the resulting embedding.
        3. Re-cluster each cluster from the first pass if needed.
        4. Combine and return final labels with unique cluster IDs, run time, etc.
        
        Args:
            data: Your high-dimensional input data (numpy array or similar).
            min_desired_clusters: Use this as a threshold for deciding to keep sub-clustering 
                                until you get enough clusters (optional logic).
            subclustering_eps: DBSCAN eps parameter for the second pass.
            subclustering_min_samples: DBSCAN min_samples parameter for the second pass.

        Returns:
            A dictionary with:
                - 'original_labels': The cluster labels from the first pass.
                - 'final_labels': The final cluster labels after subclustering (per-sample).
                - 'time': The total time spent clustering (approx).
                - 'num_clusters': The final number of clusters after subclustering.
                - 'embedding': The UMAP embedding (optional).
        """

        logging.info("Starting UMAP reduction.")
        start_time = time.time()
        embedding = self.umap_reducer.reduce(data)
        
        # Optionally save the embedding and model
        embedding_path = save_embedding(embedding, self.config)
        umap_model_file = f"umap_model_{int(time.time())}.joblib"
        self.umap_reducer.save(umap_model_file)
        
        logging.info(f"UMAP reduction completed. Embedding shape: {embedding.shape}")
        
        # --------------------------------------------------
        # First pass DBSCAN
        # --------------------------------------------------
        logging.info("Starting first-pass DBSCAN on UMAP embedding.")
        labels, elapsed_time_1, num_clusters_1 = self.dbscan_clusterer.run(embedding)

        logging.info(f"DBSCAN pass 1 completed: {num_clusters_1} clusters found in {elapsed_time_1:.2f} seconds.")
        
        # If we do NOT need to do subclustering, you could simply return here.
        # But let's assume we want more clusters or want to subdivide further.
        
        # --------------------------------------------------
        # Subclustering logic
        # --------------------------------------------------
        # We'll build a final_labels array. By default, copy first-pass labels.
        final_labels = np.copy(labels)
        
        # Keep track of how many clusters we already have. 
        # If DBSCAN returns cluster IDs { -1, 0, 1, 2, ... }, 
        # we want the next subcluster ID to start AFTER the max in the first pass.
        # (We skip -1_save_embedding in the max computation if you treat noise separately.)
        existing_cluster_ids = set(labels) - {-1}
        next_cluster_id = max(existing_cluster_ids) + 1 if existing_cluster_ids else 0
        
        # We’ll store each cluster’s indices to re-cluster individually.
        # cluster_to_indices is a dict: cluster_id -> list_of_sample_indices
        cluster_to_indices = {}
        for i, c in enumerate(labels):
            if c not in cluster_to_indices:
                cluster_to_indices[c] = []
            cluster_to_indices[c].append(i)

        # We'll decide how many total clusters we have ignoring noise:
        total_clusters = len(existing_cluster_ids)
        
        # Now let's re-cluster each cluster (except noise) if we want more clusters.
        for cluster_id in existing_cluster_ids:
            indices = cluster_to_indices[cluster_id]
            
            # Extract the subset of embedding for these points
            sub_embedding = embedding[indices]
            
            # --- Here you define your logic whether or not to re-cluster. ---
            # For example, we only re-cluster if we haven't reached min_desired_clusters:
            if total_clusters < min_desired_clusters:
                # Or you might have other conditions, e.g., "cluster size too large," etc.
                
                logging.info(f"Re-clustering cluster {cluster_id} with {len(indices)} points.")
                
                # Re-run DBSCAN with possibly different parameters (subclustering_*).
                sub_db = DBSCAN(metric="euclidean", eps=subclustering_eps, min_samples=subclustering_min_samples)
                sub_labels = sub_db.fit_predict(sub_embedding)
                
                # The sub_labels from DBSCAN will be -1, 0, 1, ..., but these are *local* to this sub-embedding.
                # We need to offset them by next_cluster_id to ensure they're unique.
                
                # Important: We may also get sub-noise (-1). We can keep it as -1 or offset it 
                # if you want each subcluster's noise to be unique. 
                # Typically, -1 means "noise" in DBSCAN and is often grouped or ignored globally.
                
                sub_unique = set(sub_labels) - {-1}
                num_subclusters = len(sub_unique)
                
                if num_subclusters > 1:
                    # Only do the re-labeling if we actually formed multiple subclusters.
                    for subc in sub_unique:
                        # subcluster_index is the new global cluster ID
                        subcluster_index = next_cluster_id
                        # All points that have subc get assigned subcluster_index
                        subc_mask = (sub_labels == subc)
                        
                        # Map local subcluster to global cluster ID
                        for j, mask_val in enumerate(subc_mask):
                            if mask_val:
                                original_index = indices[j]
                                final_labels[original_index] = subcluster_index
                        
                        next_cluster_id += 1
                    
                    # We increased the total cluster count by however many subclusters - 1
                    # (because we replaced 1 cluster with multiple new ones).
                    total_clusters = (total_clusters - 1) + num_subclusters
        
        # Calculate total time spent
        total_elapsed_time = (time.time() - start_time)
        
        # Count how many final clusters we have, ignoring noise (-1)
        final_cluster_ids = set(final_labels) - {-1}
        final_num_clusters = len(final_cluster_ids)
        
        logging.info(f"Final subclustering produced {final_num_clusters} clusters (excluding noise).")
        
        # Return dictionary with results
        return {
            "original_labels": labels.tolist(),      # From the first pass
            "final_labels": final_labels.tolist(),   # After subclustering
            "time": total_elapsed_time,
            "num_clusters": final_num_clusters,
            "embedding": embedding.tolist(),         # If you want the actual embedding returned
            "umap_model_file": umap_model_file
        }


    def run_umap_hdbscan_with_subclustering(self,
                                            data, 
                                            min_desired_clusters=1000000,
                                            subclustering_min_cluster_size=5,
                                            subclustering_min_samples=None):
        """
        1. Apply UMAP dimensionality reduction.
        2. Run HDBSCAN on the resulting embedding.
        3. (Optional) Re-cluster each cluster from the first pass if needed 
           until the total number of clusters >= min_desired_clusters (or any other criteria).
        4. Combine and return final labels with unique cluster IDs, run time, etc.

        Args:
            data (array-like): High-dimensional input data for clustering.
            min_desired_clusters (int): Threshold for deciding to keep sub-clustering.
            subclustering_min_cluster_size (int): HDBSCAN min_cluster_size for the second pass.
            subclustering_min_samples (int|None): HDBSCAN min_samples for the second pass.
                                                  If None, defaults to the same as min_cluster_size internally.

        Returns:
            A dictionary with:
                - 'original_labels': The cluster labels from the first HDBSCAN pass.
                - 'final_labels': The final cluster labels after subclustering.
                - 'time': The total time spent clustering (approx).
                - 'num_clusters': The final number of clusters (excluding noise).
                - 'embedding': The UMAP embedding (optionally returned).
                - 'umap_model_file': Path to the saved UMAP model file (if applicable).
        """
        # --------------------------------------------------
        # 1. UMAP Reduction
        # --------------------------------------------------
        logging.info("Starting UMAP reduction.")
        start_time = time.time()
        embedding = self.umap_reducer.reduce(data)

        # Optionally save the embedding and model
        embedding_path = save_embedding(embedding, self.config)
        umap_model_file = f"temp/umap_model_{int(time.time())}.joblib"
        self.umap_reducer.save(umap_model_file)

        logging.info(f"UMAP reduction completed. Embedding shape: {embedding.shape}")

        # --------------------------------------------------
        # 2. First-pass HDBSCAN
        # --------------------------------------------------
        # Adjust the parameters as desired for your first-pass HDBSCAN
        logging.info("Starting first-pass HDBSCAN on UMAP embedding.")
        labels, elapsed_time_1, num_clusters_1 = self.hdbscan_clusterer.run(embedding)


        # HDBSCAN does not directly provide runtime, so let's approximate:
        elapsed_time_1 = time.time() - start_time
        # Count how many clusters we have (excluding noise)
        unique_cluster_ids = set(labels) - {-1}
        num_clusters_1 = len(unique_cluster_ids)

        logging.info(
            f"HDBSCAN pass 1 completed: {num_clusters_1} clusters found "
            f"in {elapsed_time_1:.2f} seconds (excl. UMAP time)."
        )

        # --------------------------------------------------
        # 3. Subclustering Logic
        # --------------------------------------------------
        final_labels = np.copy(labels)
        
        # For assigning new cluster IDs during subclustering, figure out the max in the first pass
        # ignoring noise (-1).
        existing_cluster_ids = set(labels) - {-1}
        next_cluster_id = max(existing_cluster_ids) + 1 if existing_cluster_ids else 0

        # Build a map from cluster_id -> list of sample indices
        cluster_to_indices = {}
        for i, c in enumerate(labels):
            if c not in cluster_to_indices:
                cluster_to_indices[c] = []
            cluster_to_indices[c].append(i)

        # Current total cluster count (ignoring noise)
        total_clusters = len(existing_cluster_ids)

        # Re-cluster each cluster (except noise) if we have not reached our desired threshold
        for cluster_id in existing_cluster_ids:
            indices = cluster_to_indices[cluster_id]
            sub_embedding = embedding[indices]

            if total_clusters < min_desired_clusters:
                logging.info(f"Re-clustering cluster {cluster_id} with {len(indices)} points.")

                # Second-pass HDBSCAN with possibly different parameters
                sub_clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=subclustering_min_cluster_size,
                    min_samples=subclustering_min_samples,
                    metric='euclidean',
                    cluster_selection_epsilon=0.1,
                    cluster_selection_method='eom'
                )
                sub_labels = sub_clusterer.fit_predict(sub_embedding)

                # Identify the subclusters (local cluster IDs)
                sub_unique_ids = set(sub_labels) - {-1}
                if len(sub_unique_ids) > 1:
                    # Only re-assign if we formed more than 1 subcluster
                    for subc_id in sub_unique_ids:
                        global_id = next_cluster_id
                        subc_mask = (sub_labels == subc_id)
                        for j, is_in_subc in enumerate(subc_mask):
                            if is_in_subc:
                                # Map back to original sample index
                                original_index = indices[j]
                                final_labels[original_index] = global_id
                        next_cluster_id += 1

                    # Adjust total cluster count. We replaced 1 cluster with multiple new ones.
                    total_clusters = (total_clusters - 1) + len(sub_unique_ids)

        # --------------------------------------------------
        # 4. Wrap Up & Return
        # --------------------------------------------------
        total_elapsed_time = time.time() - start_time
        final_cluster_ids = set(final_labels) - {-1}
        final_num_clusters = len(final_cluster_ids)

        logging.info(f"Final subclustering produced {final_num_clusters} clusters (excluding noise).")

        return {
            "original_labels": labels.tolist(),
            "final_labels": final_labels.tolist(),
            "time": total_elapsed_time,
            "num_clusters": final_num_clusters,
            "embedding": embedding.tolist(),
            "umap_model_file": umap_model_file
        }




    def run_umap_hdbscan_with_subclustering_loop(
        self,
        data, 
        max_cluster_size: int = 50000,
        min_desired_clusters: int = 1000000,
        subclustering_min_cluster_size: int = 5,
        subclustering_min_samples: Optional[int] = None
    ):
        """
        1) Apply UMAP dimensionality reduction.
        2) Run HDBSCAN on the resulting embedding (first pass).
        3) Run multiple subclustering passes in a while loop:
             - For any cluster bigger than `max_cluster_size`, re-run HDBSCAN on that cluster alone
             - Assign new global IDs for the subclusters
             - Stop if no large clusters remain or if total clusters >= min_desired_clusters
        4) Return final labels, run time, etc.
    
        Args:
            data (array-like):
                High-dimensional input data for clustering.
            max_cluster_size (int):
                If a cluster has more than this many samples, it will be subclustered.
            min_desired_clusters (int):
                If we reach this many total clusters, subclustering stops.
            subclustering_min_cluster_size (int):
                HDBSCAN's min_cluster_size for subclustering.
            subclustering_min_samples (int|None):
                HDBSCAN's min_samples for subclustering. If None, defaults to min_cluster_size.
    
        Returns:
            dict:
                - 'original_labels': The cluster labels from the first HDBSCAN pass.
                - 'final_labels': The final cluster labels after subclustering.
                - 'time': The total time spent clustering (approx).
                - 'num_clusters': The final number of clusters (excluding noise).
                - 'embedding': The UMAP embedding (optionally returned).
                - 'umap_model_file': Path to the saved UMAP model file (if applicable).
        """
        # --------------------------------------------------
        # 1) UMAP Dimensionality Reduction
        # --------------------------------------------------

        max_cluster_size = self.config["hdbscan"]["max_cluster_size"] if self.config["hdbscan"]["max_cluster_size"] is not None else max_cluster_size
        min_desired_clusters = self.config["hdbscan"]["min_desired_clusters"] if self.config["hdbscan"]["min_desired_clusters"] is not None else min_desired_clusters
        subclustering_min_cluster_size = self.config["hdbscan"]["subclustering_min_cluster_size"] if self.config["hdbscan"]["subclustering_min_cluster_size"] is not None else subclustering_min_cluster_size
        subclustering_cluster_selection_epsilon = self.config["hdbscan"]["subclustering_cluster_selection_epsilon"] if self.config["hdbscan"]["subclustering_cluster_selection_epsilon"] is not None else subclustering_cluster_selection_epsilon
        logging.info(f"max_cluster_size: {max_cluster_size}, subclustering_cluster_selection_epsilon: {subclustering_cluster_selection_epsilon}")


        logging.info("Starting UMAP reduction.")
        start_time = time.time()
        embedding = self.umap_reducer.reduce(data)
    
        # Optionally save the embedding/model
        embedding_path = save_embedding(embedding, self.config)
        umap_model_file = f"temp/umap_model_{int(time.time())}.joblib"
        self.umap_reducer.save(umap_model_file)
    
        logging.info(f"UMAP reduction completed. Embedding shape: {embedding.shape}")
    
        # --------------------------------------------------
        # 2) First-pass HDBSCAN
        # --------------------------------------------------
        logging.info("Starting first-pass HDBSCAN on UMAP embedding.")
        # Depending on your code, this might be .run(embedding) or .fit_predict(embedding)
        labels, elapsed_time_1, num_clusters_1  = self.hdbscan_clusterer.run(embedding)  
    
        # For final cluster labels, start as a copy of the first-pass labels
        final_labels = np.copy(labels)
    
        # --------------------------------------------------
        # 3) Multiple Subclustering Passes
        # --------------------------------------------------
        # We'll do repeated passes until no big clusters remain or we reach min_desired_clusters.
        sub_pass = 0
        while True:
            sub_pass += 1
            logging.info(f"Subclustering pass #{sub_pass}...")
    
            # Determine the next available global cluster ID (ignoring noise)
            existing_cluster_ids = set(final_labels) - {-1}
            if existing_cluster_ids:
                next_cluster_id = max(existing_cluster_ids) + 1
            else:
                next_cluster_id = 0
    
            # Build a map from cluster_id -> list of sample indices
            cluster_to_indices = {}
            for i, c in enumerate(final_labels):
                if c not in cluster_to_indices:
                    cluster_to_indices[c] = []
                cluster_to_indices[c].append(i)
    
            # Check the current cluster count
            total_clusters = len(existing_cluster_ids)
    
            # If we've reached our desired cluster count, stop
            if total_clusters >= min_desired_clusters:
                logging.info(
                    f"Reached at least {min_desired_clusters} clusters, stopping subclustering."
                )
                break
    
            # Track if we split any clusters in this pass
            we_split_something = False
    
            # Re-cluster any cluster that exceeds max_cluster_size
            for cluster_id in list(existing_cluster_ids):
                if cluster_id == -1:
                    continue  # skip noise
                indices = cluster_to_indices[cluster_id]
                cluster_size = len(indices)
    
                if cluster_size > max_cluster_size:
                    logging.info(
                        f"Cluster {cluster_id} has {cluster_size} points; "
                        f"re-clustering with HDBSCAN (min_cluster_size={subclustering_min_cluster_size})."
                    )
                    sub_embedding = embedding[indices]
    
                    # Second-pass (or nth-pass) HDBSCAN with more aggressive parameters
                    sub_clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=subclustering_min_cluster_size,
                        min_samples=subclustering_min_samples,
                        metric='euclidean',
                        cluster_selection_method='leaf'
                    )
                    sub_labels = sub_clusterer.fit_predict(sub_embedding)
    
                    # Identify the subclusters (local cluster IDs)
                    sub_unique_ids = set(sub_labels) - {-1}
    
                    if len(sub_unique_ids) > 1:
                        # We replaced one big cluster with multiple subclusters
                        we_split_something = True
                        for local_id in sub_unique_ids:
                            global_id = next_cluster_id
                            subc_mask = (sub_labels == local_id)
                            for j, is_in_subc in enumerate(subc_mask):
                                if is_in_subc:
                                    original_idx = indices[j]
                                    final_labels[original_idx] = global_id
                            next_cluster_id += 1
    
                        # For samples labeled -1 in subclustering pass, 
                        # you can decide if you want to keep them as noise 
                        # or retain the original cluster_id, etc.  By default,
                        # they'll remain the same if we don't update them.
                    else:
                        # If subclustering doesn't produce multiple subclusters,
                        # keep the original cluster ID (i.e., do nothing).
                        logging.info(
                            f"Subclustering did NOT create multiple subclusters for cluster {cluster_id}. Keeping as is."
                        )
    
            # After subclustering pass, re-check how many clusters exist
            new_unique_cluster_ids = set(final_labels) - {-1}
            new_total_clusters = len(new_unique_cluster_ids)
            logging.info(
                f"After pass #{sub_pass}, total non-noise clusters = {new_total_clusters}."
            )
    
            # If we didn't split anything this pass, that means no cluster was above max_cluster_size
            # or subclustering never produced multiple clusters — we are done.
            if not we_split_something:
                logging.info(
                    f"No more clusters above {max_cluster_size} samples. Stopping subclustering."
                )
                break
    
            # Otherwise, continue to the next pass
            # (the loop checks again if total_clusters >= min_desired_clusters at the top)
    
        # --------------------------------------------------
        # 4) Wrap Up & Return
        # --------------------------------------------------
        total_elapsed_time = time.time() - start_time
        final_cluster_ids = set(final_labels) - {-1}
        final_num_clusters = len(final_cluster_ids)
    
        logging.info(
            f"Final subclustering produced {final_num_clusters} clusters (excluding noise) in "
            f"{total_elapsed_time:.2f} seconds total."
        )
    
        return {
            "original_labels": labels.tolist(),
            "final_labels": final_labels.tolist(),
            "time": total_elapsed_time,
            "num_clusters": final_num_clusters,
            "embedding": embedding.tolist(),
            "umap_model_file": umap_model_file
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
