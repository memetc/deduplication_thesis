import os
import numpy as np

def save_embedding(embedding, config):
    # Extract filename and nrows from the config dictionary
    filename = config['data']['filename']
    nrows = config['data']['nrows'] if config['data']['nrows'] is not None else "all_rows"

    # Extracting the last part of the filename
    filename_parts = filename.split('/')
    last_filename_part = filename_parts[-1].split('.')[0]

    embedding_second_dimension = embedding.shape[1] if len(embedding.shape) > 1 else 1
    
    # Creating the embeddings directory if it doesn't exist
    embeddings_dir = 'embeddings'
    os.makedirs(embeddings_dir, exist_ok=True)

    # Constructing the new filename
    new_filename = f"{last_filename_part}_{nrows}_{embedding_second_dimension}.npy"
    
    # Saving the embedding vector
    embedding_filepath = os.path.join(embeddings_dir, new_filename)
    np.save(embedding_filepath, embedding)

    print(f"Embedding vector saved successfully at: {embedding_filepath}")

