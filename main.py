import os
import logging
import json
import traceback

import numpy as np
from config import config
from pipeline.data_loader import DataLoader
from pipeline.clustering_pipeline import ClusteringPipeline
from pipeline.evaluation import evaluate_clustering
from data_loader import load_and_process_data  
from slack_utils import send_slack_message

# Configure logging as early as possible.
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def main():
    # --- Data Preparation ---
    df_test = load_and_process_data(config)
    logging.info("Loaded DataFrame with %d rows", len(df_test))
    
    # Extract extra columns needed for evaluation:
    # raw_file, scan_number, and duplicate_id (true labels)
    eval_info = df_test[['raw_file', 'scan_number', 'duplicate_id']].copy()
    
    logging.info("Loading data from DataFrame...")
    data_array = DataLoader(column_name=config["data"]["column_name"]).load_data(df_test)
    logging.info(f"Data shape: {data_array.shape}")

    # --- Run a Single Example (UMAP + HDBSCAN) ---
    pipeline = ClusteringPipeline(config)
    single_example = pipeline.run_umap_hdbscan(data_array)
    
    # Add extra columns for later evaluation.
    single_example["raw_file"] = eval_info["raw_file"].tolist()
    single_example["scan_number"] = eval_info["scan_number"].tolist()
    single_example["duplicate_id"] = eval_info["duplicate_id"].tolist()
    
    # Save the result to a new file name
    pipeline.save_results(single_example, "umap_hdbscan_all_result.json")
    logging.info("Single example result saved to umap_hdbscan_all_result.json")
    
    # --- Evaluation ---
    true_labels = eval_info["duplicate_id"].values
    predicted_labels = np.array(single_example["labels"])
    eval_metrics = evaluate_clustering(true_labels, predicted_labels)
    logging.info("Evaluation metrics:\n%s", json.dumps(eval_metrics, indent=2))
    
    # --- Send Slack Message with Evaluation Metrics ---
    message = (
        "Clustering pipeline completed successfully.\n"
        "Evaluation metrics:\n" + json.dumps(eval_metrics, indent=2)
    )
    send_slack_message(message)
    logging.info("Slack notification sent with evaluation metrics.")
    
    return single_example

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_message = (
            "Clustering pipeline encountered an error:\n" + 
            str(e) + "\n" + traceback.format_exc()
        )
        logging.error(error_message)
        try:
            send_slack_message(error_message)
            logging.info("Slack notification sent with error message.")
        except Exception as slack_e:
            logging.error("Failed to send Slack error message: %s", str(slack_e))
        raise
