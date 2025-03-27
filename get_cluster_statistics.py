import os
import json 

from deduplication_utils import process_json_file


def main():
    root_path = "/cmnfs/home/students/m.celimli/clustering/deduplication_thesis/results"
    output_json_path = os.path.join(root_path, "consolidated_results.json")

    # 1) Try loading existing consolidated results if the JSON file exists
    if os.path.exists(output_json_path):
        with open(output_json_path, 'r') as f:
            consolidated_results = json.load(f)
    else:
        consolidated_results = {}

    # 2) Gather the new JSON files in the directory
    pickle_file_paths = [
        os.path.join(root_path, f)
        for f in os.listdir(root_path)
        if f.endswith(".pkl") and f != "consolidated_results.json"
    ]

    # 3) Convert the keys that are already in consolidated_results to a set, 
    #    so we know which filenames are already processed
    processed_files = {record['pkl_file'] for record in consolidated_results.values() if 'pkl_file' in record}

    # 4) Loop through each JSON path, skip if already processed
    for idx, json_path in enumerate(pickle_file_paths, start=1):
        filename = os.path.basename(json_path)
        
        if filename in processed_files:
            print(f"Skipping already processed file: {filename}")
            continue

        # Otherwise process and update consolidated_results
        print(f"Processing new file: {filename}")
        process_json_file(json_path, idx, consolidated_results)

    # 5) Finally save the updated consolidated_results
    with open(output_json_path, 'w') as out_f:
        json.dump(consolidated_results, out_f, indent=2)

    print(f"Results written to {output_json_path}")

if __name__ == "__main__":
    main()
