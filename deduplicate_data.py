#!/usr/bin/env python
import os
import json
import logging
import pandas as pd

from .deduplication_utils import filter_by_cluster_ratio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Path to your JSON result file
    # json_result_path = "/cmnfs/home/students/m.celimli/clustering/deduplication_thesis/results/umap_hdbscan_subclustering_loop_no_aug_no_mod_train_dedup_1741374625.json"
    
    logger.info("Reading clustering output from %s", json_result_path)
    with open(json_result_path, "r") as f:
        clustering_output_dict = json.load(f)

    # Check if 'final_labels' or 'labels' is present
    if "final_labels" in clustering_output_dict["results"]:
        labels = clustering_output_dict["results"]["final_labels"]
        logger.info("Using 'final_labels' from clustering output.")
    else:
        labels = clustering_output_dict["results"]["labels"]
        logger.info("Using 'labels' from clustering output.")

    # Retrieve the data filename and load
    data_filename = clustering_output_dict["config"]["data"]["filename"]
    logger.info("Reading data from %s", data_filename)
    data_df = pd.read_parquet(data_filename, engine="pyarrow")
    logger.info("Read DataFrame with %d rows.", len(data_df))

    # Filter the DataFrame
    logger.info("Starting cluster-based filtering.")
    data_dev_df, remaining_labels = filter_by_cluster_ratio(data_df, labels, exclude_noise=True)
    logger.info("Filtering complete. Filtered DataFrame has %d rows.", len(data_dev_df))

    # Extract the "core" filename from the JSON path
    filename_parts = json_result_path.split("/")
    last_filename_part = filename_parts[-1].split(".")[0]

    # Prepare the output directory and file
    data_dir = f"{os.getcwd()}/deduplication_thesis/data/"
    os.makedirs(data_dir, exist_ok=True)
    output_parquet_path = os.path.join(data_dir, f"{last_filename_part}.parquet")

    # Save the results
    logger.info("Saving filtered dataset to %s", output_parquet_path)
    data_dev_df.to_parquet(output_parquet_path, engine="pyarrow", index=False)
    logger.info("Saved Parquet file successfully.")

if __name__ == "__main__":
    main()
