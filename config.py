config = {
    "data": {
        "filename": "unmod_val.parquet",
        "nrows": 500000,  # Optional: set to None if you want all rows
        "column_name": "intensities_raw"
    },
    "umap": {
        "n_components": 2,
        "n_neighbors": 5,
        "min_dist": 0.1,
        "metric": "cosine"
    },
    "hdbscan": {
        "min_cluster_size": 5,
        "min_samples": None,
        "metric": "euclidean"
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
    }
}
