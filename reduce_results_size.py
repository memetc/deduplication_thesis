import os
import json
import pickle

def process_json_files(directory):
    # Iterate through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)

            # Load JSON data
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Check and modify data if 'results' key exists and is a dict
            if 'results' in data and isinstance(data['results'], dict):
                modified = False
                if 'raw_file' in data['results'] or 'scan_number' in data['results'] or 'duplicate_id' in data['results']:
                    data['results'].pop('raw_file', None)
                    data['results'].pop('scan_number', None)
                    data['results'].pop('duplicate_id', None)

                    modified = True

                # Save modified data back to the same file if changes were made
                if modified:
                    with open(filepath, 'w', encoding='utf-8') as file:
                        json.dump(data, file, ensure_ascii=False, indent=2)

    print("JSON files have been updated successfully.")

def convert_json_to_pickle(directory):
    # Iterate through each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            json_filepath = os.path.join(directory, filename)
            pickle_filepath = os.path.join(directory, filename.replace('.json', '.pkl'))

            # Load JSON data
            with open(json_filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)

            # Save data in pickle format
            with open(pickle_filepath, 'wb') as pickle_file:
                pickle.dump(data, pickle_file)

            # Remove the original JSON file
            os.remove(json_filepath)

    print("JSON files have been successfully converted to pickle format and original JSON files removed.")


if __name__ == '__main__':
    convert_json_to_pickle('results')
    # process_json_files('results')