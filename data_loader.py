import pandas as pd

from preprocess_utils import assign_duplicate_ids

def load_and_process_data(config):
    """
    Loads a parquet file using a filename from the config, selects a subset of rows,
    and assigns duplicate IDs based on selected columns.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Expected to contain:
          - config["data"]["filename"]: the file path to the parquet file.
          - config["data"]["nrows"]: the number of rows to process (optional).

    Returns
    -------
    df_processed : pd.DataFrame
        The processed DataFrame with duplicate IDs assigned.
    """
    # Get filename and number of rows from the configuration
    filename = config["data"]["filename"]
    nrows = config["data"].get("nrows", None)
    
    # Read the parquet file
    df = pd.read_parquet(filename, engine="pyarrow")
    
    # Optionally select a subset of rows
    if nrows is not None:
        df = df.iloc[:nrows]
    
    # List of columns to exclude from duplicate id assignment
    columns_to_exclude = [
        'raw_file', 'scan_number', 'masses_raw', 'intensities_raw',
        'precursor_charge_onehot', 'andromeda_score', 'method_nbr',
        'unmod_sequence', 'base_intensity', 'total_intensity'
    ]
    
    # Include all columns not in the exclusion list
    columns_to_include = [col for col in df.columns if col not in columns_to_exclude]
    
    # Process the DataFrame using assign_duplicate_ids (assumed to be defined elsewhere)
    df_processed = assign_duplicate_ids(df, columns_to_include)
    
    return df_processed
