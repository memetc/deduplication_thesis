import os
import json
import joblib
import umap.umap_ as umap
import umap.plot
import numpy as np
import logging

class UMAPReducer:
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", embedding_file=None):
        """
        Initialize the UMAP reducer.

        Args:
            n_components (int): Dimensionality of the UMAP embedding.
            n_neighbors (int): The size of local neighborhood (in terms of number of neighboring sample points).
            min_dist (float): The effective minimum distance between embedded points.
            metric (str): The metric to use for the distance computation.
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.embedding_file = embedding_file
        # Create a UMAP reducer instance (not yet fit to data).
        self.reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=self.n_components,
            metric=self.metric,
        )

    def reduce(self, data):
        """
        Perform dimensionality reduction on the data using UMAP.
        If a saved UMAP model file exists, it will be loaded and only a transform
        operation will be done. Otherwise, a new model will be fit.

        Args:
            data (array-like): High-dimensional data to be reduced.

        Returns:
            embedding (ndarray): The 2D (or n_components-D) embedding of the input data.
        """
        model_path = "umap_model.joblib"

        # If there's a saved model, load and transform.
        # if os.path.exists(model_path):
        #     self.reducer = joblib.load(model_path)
        #     embedding = self.reducer.transform(data)
        #     print(f"Loaded existing UMAP model from '{model_path}' and transformed data.")
        if self.embedding_file:
            embedding = np.load(self.embedding_file)
            logging.info(f"Loaded existing UMAP embedding from '{self.embedding_file}'")
        else:
            # Fit-transform a new model on the data.
            embedding = self.reducer.fit_transform(data)
            logging.info("No existing UMAP model found. Fit a new model and performed reduction.")

        return embedding

    def save(self, filename="umap_model.joblib"):
        """
        Saves the fitted UMAP model to disk.

        Args:
            filename (str): Path to the file where the model will be saved.
        """
        joblib.dump(self.reducer, filename)
        logging.info(f"UMAP model saved to '{filename}'")

    
    def plot_embedding(self, labels=None, show=True, save_path=None):
        """
        Plot the points from the fitted UMAP reducer using UMAP's built-in plotting.
        This requires installing the plotting dependencies: pip install umap-learn[plot]
    
        Args:
            labels (array-like, optional): Labels to color the points (e.g., cluster labels).
            show (bool): Whether to display the plot immediately.
            save_path (str, optional): If provided, the plot will be saved to the given path.
        """
        try:
            import umap.plot
        except ImportError:
            raise ImportError(
                "UMAP plotting is not installed. Install with: pip install umap-learn[plot]"
            )
    
        # `umap.plot.points` uses the UMAP reducer directly.
        p = umap.plot.points(self.reducer, labels=labels)
    
        if save_path:
            # Save the plot to the given path
            p.save(save_path)
    
        if show:
            umap.plot.show(p)

