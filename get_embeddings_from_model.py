import tensorflow as tf
import numpy as np
import os
import wandb
import pandas as pd

from dlomix.data import FragmentIonIntensityDataset
from models.prosit_transformer import PrositTransformer
from dlomix.losses import masked_spectral_distance
from dlomix.reports.postprocessing import normalize_intensity_predictions
from tensorflow.keras.models import load_model
from pipeline.umap_reducer import UMAPReducer



_ALPHABET_UNMOD_ORDERED = "ACDEFGHIKLMNPQRSTVWY"
ALPHABET = {k: v for v, k in enumerate(_ALPHABET_UNMOD_ORDERED, start=1)}
ALPHABET.pop("C", None)  
ALPHABET["C[UNIMOD:4]"] = 2
ALPHABET["M[UNIMOD:35]"] = len(ALPHABET) + 1



def extract_embeddings(config, train_data, project_name, layer_name="transformer_encoder_1"):
    """
    Loads the model specified by `config["saved_best_weights"]` and extracts
    embeddings from `layer_name` for the test data.

    Args:
        config: dict-like, containing the paths and hyperparams, e.g. config["saved_best_weights"].
        project_name: str, name of the W&B project for logging.
        layer_name: str, name of the sub-layer from which to extract embeddings.

    Returns:
        A DataFrame containing the sequences and their corresponding embeddings.    
    """


    with wandb.init(config=config, project=project_name,  dir="/cmnfs/proj/prosit/dedup/") as _:
        config = wandb.config
        config = dict(wandb.config)
        print(config)
        print("____________")
        assert "data_source" in config
        assert "train" in config["data_source"]
        model = load_model(config["saved_best_weights"], compile=False)

        outputs, embeddings = model.predict(train_data, batch_size=config["batch_size"])
    
        return embeddings
    



def get_proteometools_data(config):
    train_path, val_path = config["data_source"].values()
    BATCH_SIZE = config["batch_size"]
    
    int_data = FragmentIonIntensityDataset(
    data_source=train_path,
    val_data_source=val_path,
    disable_cache=False,
    data_format="parquet", 
    sequence_column="modified_sequence",
    label_column="intensities_raw",
    encoding_scheme = "naive-mods", 
    max_seq_len=30,
    batch_size=BATCH_SIZE,
    model_features=[
        "precursor_charge_onehot",
        "collision_energy_aligned_normed",
        "method_nbr",
    ],
    alphabet=ALPHABET,
    with_termini=False,
    num_proc=8, # cpus-per-task
    )

    return int_data.tensor_train_data, int_data
        

if __name__ == "__main__":

    
    config = {
    "batch_size": 8192,
    "data_source": {
        "train": "/cmnfs/home/students/m.celimli/clustering/deduplication_thesis/data/no_aug_no_mod_train_dedup.parquet",
        "val": "/cmnfs/home/students/m.celimli/clustering/deduplication_thesis/data/no_aug_no_mod_val_dedup.parquet"
    },
    "saved_model_location": "/cmnfs/proj/prosit/dedup",
    "name_run": "no_aug_no_mod_dedupped_dlomix",
    "dense_dim_factor": 4,
    "dropout_rate": 0,
    "early_stopping": {
        "min_delta": 0.0001,
        "patience": 8
    },
    "embedding_output_dim": 64,
    "epochs": 200,
    "ff_dim": 32,
    "learning_rate": 0.0001,
    "len_fion": 6,
    "num_heads": 16,
    "num_transformers": 6,
    "reduce_lr": {
        "factor": 0.8,
        "patience": 4
    },
    "transformer_dropout": 0.1,
    "model_class": "PrositTransformer",
    "saved_best_weights": "/cmnfs/proj/prosit/dedup/no_aug_no_mod_train_dedup.keras"
    }
    
    umap_reducer = UMAPReducer(
    n_components=2,
    n_neighbors=5,
    min_dist=0.1,
    metric="euclidean",
    embedding_file = None
    )


    train_data, int_data = get_proteometools_data(config)
    modal_embeddings = extract_embeddings(config, train_data, "transformer-baseline-deduplication", 'transformer_encoder_1')
    np.save("embeddings/train_dedup_raw_embeddings_from_model_pca2.npy", modal_embeddings)


    embedding_path = 'embeddings/train_dedup_raw_embeddings_from_model_pca2.npy'
    modal_embeddings = np.load(embedding_path)
    modal_embeddings_2d = umap_reducer.reduce(modal_embeddings)
    np.save("embeddings/train_dedup_embeddings_from_model_2_euclidean.npy", modal_embeddings_2d)
