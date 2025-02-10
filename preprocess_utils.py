import pandas as pd

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
    df_copy = df.copy()  # Avoid modifying the original DataFrame

    # Generate unique group IDs based on duplicates
    df_copy["duplicate_id"] = df_copy.groupby(columns_to_include).ngroup()

    # Count the number of duplicates for each group
    df_copy["duplicate_count"] = df_copy.groupby("duplicate_id")["duplicate_id"].transform("count")

    return df_copy


def drop_hcd_itms_rows(df):
    """
    Drops rows in 'df' where fragmentation == 'HCD' 
    and mass_analyzer == 'ITMS', then resets the index.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to modify.

    Returns:
    --------
    pandas.DataFrame
        The modified DataFrame with rows dropped and index reset.
    """
    # Create a boolean mask for rows to drop
    mask = (df["fragmentation"] == "HCD") & (df["mass_analyzer"] == "ITMS")

    # Drop the matching rows in place
    df.drop(df[mask].index, inplace=True)

    # Reset the index, discarding the old index
    df.reset_index(drop=True, inplace=True)

    return df


def keep_highest_andromeda_score(df):
    """
    For each unique duplicate_id in the DataFrame, keep only the row 
    with the highest andromeda_score and remove the rest.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing columns 'duplicate_id' and 'andromeda_score'.

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each unique duplicate_id appears once, 
        with the highest andromeda_score retained.
    """

    # Option 1 (Sorting + dropping duplicates):
    # ----------------------------------------
    # 1) Sort by 'andromeda_score' descending (so highest is first).
    # 2) Drop duplicates by 'duplicate_id' keeping the first occurrence
    #    (i.e., the highest score).
    # 3) (Optional) Reset index if desired.

    df_sorted = (
        df.sort_values("andromeda_score", ascending=False)
          .drop_duplicates(subset="duplicate_id", keep="first")
          .reset_index(drop=True)
    )

    return df_sorted

