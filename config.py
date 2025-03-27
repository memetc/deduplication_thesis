config = {
    "data": {
        "filename": "/cmnfs/home/students/m.celimli/clustering/deduplication_thesis/data/no_aug_no_mod_train_dedup.parquet",
        #"filename": "/cmnfs/proj/prosit/Transformer/no_aug_no_mod_train.parquet",
        "nrows": None,  # Optional: set to None if you want all rows
        "column_name": "intensities_raw"
    },
    "umap": {
        "n_components": 30,
        "n_neighbors": 5,
        "min_dist": 0.5,
        "metric": "euclidean",
        #"embedding_file": None
        # "embedding_file": "/cmnfs/home/students/m.celimli/clustering/deduplication_thesis/embeddings/train_dedup_embeddings_from_model.npy" 
         "embedding_file": "/cmnfs/home/students/m.celimli/clustering/deduplication_thesis/embeddings/no_aug_no_mod_train_dedup_all_rows_30.npy" 
    },
    "hdbscan": {
        "min_cluster_size": 3,
        "min_samples": None,
        "metric": "euclidean",
        "cluster_selection_epsilon": 0.1,
        "cluster_selection_method": "eom",
        "max_cluster_size": 100,
        "min_desired_clusters": 1000000,
        "subclustering_min_cluster_size": 5,
        "subclustering_cluster_selection_epsilon": 0.05
    },
    "knn": {
        "k": 3,
        "use_sqrt": True
    },
    "sparse_hdbscan": {
        "min_cluster_size": 3
    },
    "leiden": {
        "resolution": 5.0
    },
    "sparse_dbscan": {
        "eps": 0.10,
        "min_samples": 3,
        "subclustering_min_samples":3,
        "subclustering_eps": 0.08
    },
    "birch":
    {
        "threshold":0.05,
        "branching_factor": 50

    }
}
