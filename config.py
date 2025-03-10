config = {
    "data": {
        #"filename": "/cmnfs/home/students/m.celimli/clustering/deduplication_thesis/data/no_aug_no_mod_train_dedup.parquet",
        "filename": "/cmnfs/proj/prosit/Transformer/no_aug_no_mod_val.parquet",
        "nrows": None,  # Optional: set to None if you want all rows
        "column_name": "intensities_raw"
    },
    "umap": {
        "n_components": 30,
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
        "eps": 0.2,
        "min_samples": 5
    }
}
