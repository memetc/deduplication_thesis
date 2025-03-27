#!/usr/bin/env python
import os
import logging
import json
import pickle
import traceback
import numpy as np
import time

from config import config
from pipeline.data_loader import DataLoader
from pipeline.clustering_pipeline import ClusteringPipeline
from pipeline.evaluation import evaluate_clustering
from data_loader import load_and_process_data
from slack_utils import send_slack_message

# Configure logging as early as possible.
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def main():
    # --- Notify Slack that the run has started ---
    start_message = "UMAP+HDBSCAN subclustering run has started."
    send_slack_message(start_message)
    logging.info("Sent Slack notification: %s", start_message)
    
    # --- Data Preparation ---
    df_test = load_and_process_data(config)
    logging.info("Loaded DataFrame with %d rows", len(df_test))
    
    # Extract extra columns needed for evaluation
    eval_info = df_test[['raw_file', 'scan_number', 'duplicate_id']].copy()
    
    logging.info("Loading data from DataFrame...")
    data_array = DataLoader(column_name=config["data"]["column_name"]).load_data(df_test)
    logging.info(f"Data shape: {data_array.shape}")
    
    # --- Run UMAP + HDBSCAN (with subclustering) ---
    pipeline = ClusteringPipeline(config)
    # Make sure your ClusteringPipeline class has this method implemented:
    #    run_umap_hdbscan_with_subclustering(data)
    result = pipeline.run_umap_hdbscan_with_subclustering(data_array)
    
    # If 'final_labels' is where your clustering IDs reside, but you want the rest of the
    # script (evaluation, saving) to treat them as "labels", map it here:
    result["labels"] = result["final_labels"]
    
    # Append extra columns for evaluation.
    # result["raw_file"] = eval_info["raw_file"].tolist()
    # result["scan_number"] = eval_info["scan_number"].tolist()
    # result["duplicate_id"] = eval_info["duplicate_id"].tolist()

    filename = config['data']['filename']
    filename_parts = filename.split('/')
    last_filename_part = filename_parts[-1].split('.')[0]

    # Save results to file
    out_filename = f"results/umap_hdbscan_subclustering_{last_filename_part}_{int(time.time())}.pkl"
    pipeline.save_results(result, out_filename)
    logging.info(f"Result saved to {out_filename}")
    
    # --- Evaluation ---
    true_labels = eval_info["duplicate_id"].values
    predicted_labels = np.array(result["labels"])
    eval_metrics = evaluate_clustering(true_labels, predicted_labels)
    logging.info("Evaluation metrics:\n%s", json.dumps(eval_metrics, indent=2))
    
    # --- Send Slack Message with Evaluation Metrics ---
    message = (
        "UMAP+HDBSCAN (with subclustering) completed successfully.\n"
        "Evaluation metrics:\n" + json.dumps(eval_metrics, indent=2)
    )
    send_slack_message(message)
    logging.info("Slack notification sent with evaluation metrics.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_message = (
            "UMAP+HDBSCAN subclustering pipeline encountered an error:\n" +
            str(e) + "\n" + traceback.format_exc()
        )
        logging.error(error_message)
        try:
            send_slack_message(error_message)
            logging.info("Slack error notification sent.")
        except Exception as slack_e:
            logging.error("Failed to send Slack error message: %s", str(slack_e))
        raise
