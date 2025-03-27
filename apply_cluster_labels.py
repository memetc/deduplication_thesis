import pickle
import os
import time
import json
import pandas as pd
import numpy as np

from deduplication_utils import filter_by_cluster_ratio, filter_by_cluster_and_sequence_ratio


def load_clustering_results(json_result_path):
    """Load the clustering results from a JSON file."""
    with open(json_result_path, 'r') as file:
        clustering_output = pickle.load(file)
    return clustering_output

def load_dataset(parquet_path):
    """Load a dataset from a parquet file."""
    return pd.read_parquet(parquet_path, engine="pyarrow")

def assign_duplicate_ids(df, columns_to_exclude=None):
    """
    Assigns a unique duplicate ID and counts duplicates.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns_to_exclude (list, optional): List of columns to exclude 
            from duplicate determination.
    
    Returns:
        pd.DataFrame: DataFrame with 'duplicate_id' and 'duplicate_count' columns.
    """
    df = df.copy()
    if columns_to_exclude is None:
        columns_to_exclude = ['raw_file', 'scan_number', 'masses_raw', 
                              'intensities_raw', 'andromeda_score', 'precursor_charge_onehot']
    columns_to_include = [col for col in df.columns if col not in columns_to_exclude]
    df["duplicate_id"] = df.groupby(columns_to_include).ngroup()
    df["duplicate_count"] = df.groupby("duplicate_id")["duplicate_id"].transform("count")
    return df

def merge_cluster_labels(main_df, clustering_df, merge_cols=["raw_file", "scan_number"]):
    """
    Merge cluster labels onto the main dataframe based on merge columns.
    
    Parameters:
        main_df (pd.DataFrame): The primary dataset.
        clustering_df (pd.DataFrame): The clustering labels dataframe.
        merge_cols (list): List of columns to merge on.
    
    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    merged_df = pd.merge(main_df, clustering_df, how='left', on=merge_cols)
    return merged_df

def fill_cluster_labels_dict(df, id_col="duplicate_id", label_col="clustering_labels"):
    """
    Fills NaN cluster labels within duplicate groups using the first non-NaN label.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing duplicate groups and cluster labels.
        id_col (str): Column name used for grouping.
        label_col (str): Column name with the cluster labels.
    
    Returns:
        pd.DataFrame: DataFrame with filled cluster labels.
    """
    df = df.copy()
    mapping_df = df.dropna(subset=[label_col]).drop_duplicates(subset=[id_col])
    id_to_label = dict(zip(mapping_df[id_col], mapping_df[label_col]))
    df[label_col] = df[label_col].fillna(df[id_col].map(id_to_label))
    return df


def apply_cluster_labels(input_parquet, clustering_json, clustering_subset_cols=["raw_file", "scan_number", "clustering_labels"]):
    """
    Apply clustering labels from a clustering JSON to a dataset.
    
    Parameters:
        input_parquet (str): Path to the input dataset (parquet file).
        clustering_json (dict): Clustering result dictionary loaded from JSON.
        clustering_subset_cols (list): Columns from the clustering dataset to merge.
    
    Returns:
        pd.DataFrame: The dataset with cluster labels applied and filled.
    """
    # Load the main dataset
    df_main = load_dataset(input_parquet)
    print(1, len(df_main))

    # Assign duplicate IDs based on all columns except a specified set
    df_main = assign_duplicate_ids(df_main)
    
    # Load the clustering subset data and attach the cluster labels
    clustering_data_filename = clustering_json['config']['data']['filename']
    df_clustering = load_dataset(clustering_data_filename)
    df_clustering['clustering_labels'] = clustering_json['results']['final_labels']
    df_clustering = df_clustering[clustering_subset_cols]
    print(2, len(df_clustering))

    # Merge clustering labels with the main dataset
    merged_df = merge_cluster_labels(df_main, df_clustering)
    print(3, len(merged_df))

    # Fill missing cluster labels using the duplicate group strategy
    merged_df = fill_cluster_labels_dict(merged_df)
    print(4, len(merged_df))
    labels = merged_df['clustering_labels'].tolist()

    # Drop intermediate columns not needed for further analysis
    merged_df.drop(columns=["duplicate_id", "duplicate_count", "clustering_labels"], inplace=True, errors="ignore")
    print(5, len(merged_df))

    return merged_df, labels

def save_dataset(df, output_dir, prefix="dedupped"):
    """
    Save the DataFrame to a parquet file with a timestamped filename.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        output_dir (str): Directory to save the file.
        prefix (str): Filename prefix.
    
    Returns:
        str: The full path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{prefix}.parquet")
    df.to_parquet(output_file, engine="pyarrow", index=False)
    return output_file

# Example usage:
if __name__ == "__main__":
    # Paths for clustering JSON and the input dataset
    filename = "umap_dbscan_subclustering_no_aug_no_mod_train_dedup_1742474510"
    ratio = 40

    clustering_json_path = f'/cmnfs/home/students/m.celimli/clustering/deduplication_thesis/results/{filename}.pkl'
    input_dataset_path = "/cmnfs/proj/prosit/Transformer/no_aug_no_mod_train.parquet"
    output_dir = os.path.join(os.getcwd(), "data/")
    prefix = f'{filename}_applied_on_all_{ratio}'

    # Load the clustering results
    clustering_json = load_clustering_results(clustering_json_path)
    
    # Apply the cluster labels to the dataset
    result_df, labels = apply_cluster_labels(input_dataset_path, clustering_json)
    
    print(len(result_df))
    # Optionally, you can filter the dataset based on cluster ratio
    result_df_filtered, _ = filter_by_cluster_ratio(result_df, labels, ratio=ratio/100)
    print(len(result_df_filtered))

    # Save the resulting dataset
    output_file = save_dataset(result_df_filtered, output_dir, prefix=prefix)
    print("Saved re-labeled dataset to:", output_file)
