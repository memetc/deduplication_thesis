config = {
    "data": {
        "filename": "/cmnfs/data/proteomics/Prosit_unmod/intensity/unmod_val.parquet",
        "nrows": None,  # Optional: set to None if you want all rows
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
    },
    "sparse_dbscan": {
        "eps": 0.5,
        "min_samples": 5
    }
}
