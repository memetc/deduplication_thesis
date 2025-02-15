
import os
import glob
import logging
import pandas as pd
import traceback
from typing import List
from preprocess_utils import assign_duplicate_ids, drop_hcd_itms_rows, keep_highest_andromeda_score
from slack_utils import send_slack_message

def tidy_up_prosit_data(parquet_files: List[str]) -> None:
    """
    Reads each .parquet file from the given directory_path, applies a series of 
    cleaning and deduplication functions, and writes out a '_deduplicated_simple.parquet' file.
    """

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    columns_to_exclude = [
	    'raw_file',
	    'scan_number',
	    'masses_raw',
	    'intensities_raw',
	    'precursor_charge_onehot',
	    'andromeda_score',
	    'method_nbr',
	    'unmod_sequence',
	    'base_intensity',
	    'total_intensity'
	]
		
    for file_path in parquet_files:
        logger.info(f"Reading file: {file_path}")
        df = pd.read_parquet(file_path, engine="pyarrow")
        logger.info(f"Initial length: {len(df)}")
        
        send_slack_message(f'{df.columns}')
        columns_to_include = [col for col in df.columns if col not in columns_to_exclude]
        # 1) Assign duplicate IDs
        df = assign_duplicate_ids(df, columns_to_include)
        logger.info(f"After assign_duplicate_ids: {len(df)}")
        send_slack_message(f'{df.columns}')
        # 2) Drop HCD ITMS rows
        #df = drop_hcd_itms_rows(df)
        #logger.info(f"After drop_hcd_itms_rows: {len(df)}")

        # 3) Keep highest Andromeda score
        df = keep_highest_andromeda_score(df)
        logger.info(f"After keep_highest_andromeda_score: {len(df)}")

        # Write out the processed DataFrame
        output_file_path = file_path.replace('.parquet', '_deduplicated_simple.parquet')
        df.to_parquet(output_file_path, engine="pyarrow")
        logger.info(f"Saved deduplicated file: {output_file_path} (length: {len(df)})\n")
        message = f"Saved deduplicated file: {output_file_path} (length: {len(df)})\n"
        send_slack_message(message)

if __name__ == "__main__":
    try:
        tidy_up_prosit_data(parquet_files=["/cmnfs/proj/prosit/Transformer/no_aug_no_mod_train.parquet",
                                           "/cmnfs/proj/prosit/Transformer/no_aug_no_mod_val.parquet"])
    except Exception as e:
        error_message = (
            "Simple deduplication pipeline encountered an error:\n" + 
            str(e) + "\n" + traceback.format_exc()
        )
        logging.error(error_message)
        try:
            send_slack_message(error_message)
            logging.info("Slack notification sent with error message.")
        except Exception as slack_e:
            logging.error("Failed to send Slack error message: %s", str(slack_e))
        raise
