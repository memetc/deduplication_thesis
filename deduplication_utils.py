import pandas as pd
import numpy as np
import json
import os
import logging
import matplotlib.pyplot as plt

from collections import Counter
from math import log2

# Configure the logging level and format (this could go in your main script or near the top of this file)
logging.basicConfig(
    level=logging.INFO,  # or logging.DEBUG if you want more verbose output
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def filter_by_cluster_ratio(
    df: pd.DataFrame,
    labels: np.ndarray,
    ratio: float = 0.9,
    random_state: int = 42,
    exclude_noise: bool = True
) -> (pd.DataFrame, np.ndarray):
    """
    Filters (subsamples) a specified ratio of rows from each cluster in a DataFrame.
    Returns the filtered DataFrame and a NumPy array of the corresponding cluster labels.

    Args:
        df (pd.DataFrame): The DataFrame to filter. Must have the same length as labels.
        labels (np.ndarray): Cluster labels for each row in df. Typically from DBSCAN or other algorithms.
        ratio (float): Fraction of rows to keep from each cluster (0 < ratio <= 1).
        random_state (int): Seed for reproducible sampling.
        exclude_noise (bool): If True, exclude cluster -1 (i.e., noise in DBSCAN) from the final result.

    Returns:
        filtered_df (pd.DataFrame): The subsampled DataFrame.
        filtered_labels (np.ndarray): The corresponding cluster labels for the subsampled rows.
    """
    if len(df) != len(labels):
        raise ValueError("DataFrame and labels must have the same length.")

    # Convert labels to a Series so we can group conveniently.
    labels_series = pd.Series(labels, name="_cluster_label", index=df.index)
    df_labeled = df.assign(_cluster_label=labels_series)

    # Optionally exclude noise if the label is -1.
    if exclude_noise:
        df_labeled = df_labeled[df_labeled["_cluster_label"] != -1]

    # Group by cluster label and sample the specified fraction.
    # Note: If ratio=1.0, it will return all rows for that cluster.
    grouped = df_labeled.groupby("_cluster_label", group_keys=False)
    filtered_df = (
        grouped.apply(lambda x: x.sample(frac=ratio, random_state=random_state))
        .reset_index(drop=True)
    )

    # Extract the (subsampled) labels and drop the temporary column
    filtered_labels = filtered_df["_cluster_label"].values
    filtered_df = filtered_df.drop(columns="_cluster_label")

    return filtered_df, filtered_labels

def shannon_entropy(label_list):
    """
    Calculate the Shannon entropy of the distribution of labels in label_list.
    """
    count = Counter(label_list)
    total = sum(count.values())
    # sum( p * log2(1/p) ) for each label
    entropy = 0.0
    for label_count in count.values():
        p = label_count / total
        entropy -= p * log2(p)
    return entropy

def average_group_entropy(df, sequence_col='modified_sequence', label_col='cluster_label'):
    """
    Group by 'sequence_col' and compute the weighted average entropy of 'label_col' distributions.
    
    Returns a single float: lower is better (indicates purer grouping).
    """
    # Split the dataframe into groups by modified_sequence
    groups = df.groupby(sequence_col)

    total_count = len(df)
    weighted_entropy_sum = 0.0

    # For each group, compute the entropy of its labels, then weight by group size
    for _, group_df in groups:
        group_labels = group_df[label_col].tolist()
        group_size = len(group_labels)
        e = shannon_entropy(group_labels)
        weighted_entropy_sum += (group_size / total_count) * e

    return weighted_entropy_sum


def plot_dominant_cluster_fraction(
    df, 
    data_name="data",
    result_index=1,
    sequence_col='modified_sequence',
    cluster_col='cluster_label'
):
    """
    Create and save a histogram of the fraction of the dominant cluster in each group.
    The figure is saved under /cmnfs/home/students/m.celimli/clustering/deduplication_thesis/results/images
    with a name that includes 'data_name' and 'result_index'.
    """
    grouped = df.groupby(sequence_col)[cluster_col]
    
    fractions = []
    for _, labels in grouped:
        counter = Counter(labels)
        most_common_count = counter.most_common(1)[0][1]
        total_count = sum(counter.values())
        fraction = most_common_count / total_count
        fractions.append(fraction)
    
    plt.figure()
    plt.hist(fractions, bins=50)  # adjust bins if needed
    plt.xlabel('Dominant cluster fraction')
    plt.ylabel('Number of sequences')
    plt.title(f'Histogram of dominant cluster fraction per {sequence_col}')
    plt.tight_layout()

    # Ensure the images directory exists
    images_dir = "/cmnfs/home/students/m.celimli/clustering/deduplication_thesis/results/images"
    os.makedirs(images_dir, exist_ok=True)

    # Build the output plot path
    output_plot_path = os.path.join(
        images_dir,
        f"dominant_cluster_fraction_{data_name}_{result_index}.png"
    )

    # Save the figure
    plt.savefig(output_plot_path)
    plt.close()

def process_json_file(json_path, result_index, output_dict): 
    """
    Reads a single JSON file, extracts relevant data, determines
    the clustering algorithm, and computes average entropy.
    It then appends the result to output_dict, including saving the
    JSON file name.
    """
    logging.info(f"Processing JSON file: {json_path}")

    with open(json_path, 'r') as f:
        file_content = json.load(f)

    # Save the JSON file name for reference.
    json_filename = os.path.basename(json_path)

    # Extract data filename from config
    data_filename = os.path.basename(file_content["config"]["data"]["filename"])
    data_name = os.path.splitext(data_filename)[0]

    # Append json_filename to data_name for plotting purposes
    plot_data_name = json_filename

    # Determine the clustering algorithm from the file name or content
    if "hdbscan" in json_path.lower():
        clustering_alg = "hdbscan"
        args = file_content["config"]["hdbscan"]
    elif "dbscan" in json_path.lower():
        clustering_alg = "dbscan"
        args = file_content["config"]["sparse_dbscan"]
    elif "birch" in json_path.lower():
        clustering_alg = "birch"
        args = file_content["config"].get("birch", {})
    else:
        clustering_alg = "unknown"
        args = {}

    logging.info(f"Detected data name: {data_name}")
    logging.info(f"Detected clustering algorithm: {clustering_alg}, args: {args}")

    # Load labels
    results_data = file_content['results']
    logging.info(f"{file_content['results'].keys()}")

    labels = results_data.get('final_labels', results_data.get('labels'))
    labels = np.array(labels)

    # Count noise and unique clusters
    noise_count = np.sum(labels == -1)
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    num_clusters = len(unique_labels)

    # Read the original data
    parquet_path = file_content['config']['data']['filename']
    data_df = pd.read_parquet(parquet_path, engine="pyarrow")

    # Calculate the average entropy
    data_df['cluster_label'] = labels
    avg_entropy = average_group_entropy(data_df)

    # Log statistics
    logging.info(f"Noise points: {noise_count}, Number of clusters (excluding noise): {num_clusters}")
    logging.info(f"Average group entropy: {avg_entropy:.4f}")

    # Generate and save the plot; include the json file name in the data_name for the plot filename.
    plot_dominant_cluster_fraction(
        data_df,
        data_name=plot_data_name,
        result_index=result_index,
        sequence_col='modified_sequence',
        cluster_col='cluster_label'
    )
    logging.info(f"Saved dominant-cluster-fraction plot for: {plot_data_name}")

    # Store results in output_dict, including the JSON file name.
    result_key = f"result-{result_index}"
    output_dict[result_key] = {
        "json_file": json_filename,
        "data": data_name,
        "clustering_alg": clustering_alg,
        "args": args,
        "avg_entropy": float(avg_entropy),
        "noise_count": int(noise_count),
        "num_clusters": int(num_clusters),
    }

    logging.info(f"Finished processing JSON file: {json_path}\n")