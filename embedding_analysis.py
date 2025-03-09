import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import Counter

#####################
# Function Definitions
#####################

def plot_embeddings_with_colors(embeddings, ids, title, filename=None):
    """
    Plots 2D embeddings with color coding based on the 'ids' array.
    Saves the figure if 'filename' is provided.
    """
    x = embeddings[:, 0]
    y = embeddings[:, 1]

    unique_ids = np.unique(ids)
    # If many unique IDs, pick a larger colormap (e.g. 'tab20' or 'nipy_spectral')
    cmap = cm.get_cmap('tab20', len(unique_ids))

    plt.figure(figsize=(8, 6))
    for i, uid in enumerate(unique_ids):
        idx = (ids == uid)
        color = cmap(i)
        plt.scatter(x[idx], y[idx], c=[color], marker='o', edgecolor='k', label=str(uid))

    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # place legend outside
    plt.grid(True)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()  # Close the figure

def assign_duplicate_ids(df, columns_to_include):
    """
    Assigns a unique ID to each duplicate group and counts the number of duplicates.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_include (list): The list of columns to consider for duplication.

    Returns:
        pd.DataFrame: A modified DataFrame with two new columns:
                      - 'duplicate_id': A unique identifier for each group of duplicates.
                      - 'duplicate_count': The number of times that group appears.
    """
    df_copy = df.copy()
    df_copy["duplicate_id"] = df_copy.groupby(columns_to_include).ngroup()
    df_copy["duplicate_count"] = df_copy.groupby("duplicate_id")["duplicate_id"].transform("count")
    return df_copy

def count_peaks(lst):
    """
    Count how many items in 'lst' are neither -1 nor 0.
    """
    return sum(1 for x in lst if x != -1 and x != 0)

def extract_and_get_length(seq):
    """
    Remove square-bracketed content, then extract the sequence between dashes (-).
    Return the extracted sequence and its length.
    """
    # Remove all square-bracketed content
    seq_clean = re.sub(r'\[.*?\]', '', seq)
    # Extract the actual sequence between dashes
    match = re.search(r'-([A-Za-z]+)-', seq_clean)
    if match:
        extracted = match.group(1)
        return extracted, len(extracted)
    return None, 0

#####################
# Main Function
#####################

def main():
    # -----------------------------
    # 1. Configuration / Paths
    # -----------------------------
    embedding_path = '/cmnfs/home/students/m.celimli/clustering/deduplication_thesis/embeddings/train_dedup_embeddings_from_model_2_euclidean.npy'
    dedup_train_filepath = '/cmnfs/home/students/m.celimli/clustering/deduplication_thesis/data/no_aug_no_mod_train_dedup.parquet'
    plots_root_dir = '/cmnfs/home/students/m.celimli/clustering/deduplication_thesis/plots'

    # Create a subdirectory under plots, named after the embeddings file
    embedding_basename = os.path.splitext(os.path.basename(embedding_path))[0]
    output_dir = os.path.join(plots_root_dir, embedding_basename)
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # 2. Load data
    # -----------------------------
    print("Loading parquet data...")
    dedup_train_df = pd.read_parquet(dedup_train_filepath, engine='pyarrow')
    print(f"Data loaded. Shape: {dedup_train_df.shape}")

    print("Loading embeddings from .npy...")
    embeddings = np.load(embedding_path)
    print(f"Embeddings loaded. Shape: {embeddings.shape}")

    # -----------------------------
    # 3. Preprocessing
    # -----------------------------
    # 3A. Derive precursor_charge_label from precursor_charge_onehot
    print("Deriving precursor_charge_label...")
    dedup_train_df["precursor_charge_label"] = dedup_train_df["precursor_charge_onehot"].apply(np.argmax)

    # 3B. Identify columns to exclude for duplicate assignment
    columns_to_exclude = [
        'raw_file', 'scan_number', 'masses_raw', 'intensities_raw',
        'andromeda_score', 'precursor_charge_onehot'
    ]
    columns_to_include = [col for col in dedup_train_df.columns if col not in columns_to_exclude]

    print("Assigning duplicate IDs...")
    dev_df = assign_duplicate_ids(dedup_train_df, columns_to_include=columns_to_include)

    # 3C. Count peaks
    print("Counting peaks...")
    dev_df['number_of_peaks'] = dev_df['intensities_raw'].apply(count_peaks)

    # 3D. Extract sequence & length
    print("Extracting sequences & computing lengths...")
    dev_df[['extracted_sequence', 'sequence_length']] = dev_df['modified_sequence'].apply(
        lambda x: pd.Series(extract_and_get_length(x))
    )

    # -----------------------------
    # 4. Plotting
    # -----------------------------
    print("Plotting with different color codings...")

    # 4A. Plot by precursor_charge_label
    ids_charge = dev_df['precursor_charge_label'].values
    charge_filename = os.path.join(output_dir, f"{embedding_basename}_2D_charge.png")
    plot_embeddings_with_colors(
        embeddings=embeddings,
        ids=ids_charge,
        title='UMAP Embeddings (Color = Charge Label)',
        filename=charge_filename
    )

    # 4B. Plot by number_of_peaks
    ids_peaks = dev_df['number_of_peaks'].values
    peaks_filename = os.path.join(output_dir, f"{embedding_basename}_2D_peaks.png")
    plot_embeddings_with_colors(
        embeddings=embeddings,
        ids=ids_peaks,
        title='UMAP Embeddings (Color = # of Peaks)',
        filename=peaks_filename
    )

    # 4C. Plot by sequence_length
    ids_seq_len = dev_df['sequence_length'].values
    seq_len_filename = os.path.join(output_dir, f"{embedding_basename}_2D_sequence_length.png")
    plot_embeddings_with_colors(
        embeddings=embeddings,
        ids=ids_seq_len,
        title='UMAP Embeddings (Color = Sequence Length)',
        filename=seq_len_filename
    )

    print(f"Plots have been saved to: {output_dir}")

# Standard Python convention to call main() when the script is run
if __name__ == "__main__":
    main()
