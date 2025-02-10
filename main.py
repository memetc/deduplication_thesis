import pandas as pd
import numpy as np
import logging
import json

from config import config
from pipeline.data_loader import DataLoader
from pipeline.umap_reducer import UMAPReducer
from pipeline.clustering_pipeline import ClusteringPipeline
from pipeline.evaluation import evaluate_clustering
from data_loader import load_and_process_data  

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def main():
    # Data Preparation

    df_test = load_and_process_data(config)
    
    print(len(df_test))
    logging.info("Loading data from DataFrame...")
    data_array = DataLoader(column_name=config["data"]["column_name"]).load_data(df_test)
    logging.info(f"Data shape: {data_array.shape}")

    true_labels = df_test["duplicate_id"].values

    # --- Run a Single Example (UMAP + HDBSCAN) ---
    pipeline = ClusteringPipeline(config)
    single_example = pipeline.run_umap_hdbscan(data_array)
    pipeline.save_results(single_example, "single_example_result.json")
    logging.info("Single example result saved to single_example_result.json")

    return single_example
    # Create and run the clustering pipeline
    pipeline = ClusteringPipeline(config)
    results = pipeline.run_all(data_array)
    pipeline.save_results(results, "clustering_results.json")

    # --- Evaluation ---
    evaluation_results = {}
    for strategy, result in results.items():
        predicted_labels = result["labels"]
        eval_metrics = evaluate_clustering(true_labels, predicted_labels)
        evaluation_results[strategy] = eval_metrics


    # Save evaluation metrics along with the clustering results
    final_output = {"config": config, "results": results, "evaluation": evaluation_results}
    with open("final_results.json", "w") as f:
        json.dump(final_output, f, indent=4)
    logging.info("Final results (with evaluation) saved to final_results.json")







