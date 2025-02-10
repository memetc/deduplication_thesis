import numpy as np
import faiss
import scipy.sparse as sp
import logging

class KNNGraph:
    def __init__(self, k=50, use_sqrt=True):
        self.k = k
        self.use_sqrt = use_sqrt

    def compute_graph(self, data):
        num_samples, dim = data.shape
        index = faiss.IndexFlatL2(dim)
        index.add(data)
        # Search for k+1 neighbors (including self)
        D, I = index.search(data, self.k + 1)
        if self.use_sqrt:
            D = np.sqrt(D)
        # Remove self-match (first column)
        D, I = D[:, 1:], I[:, 1:]
        row_indices = np.repeat(np.arange(num_samples), self.k)
        col_indices = I.flatten()
        distances = D.flatten()
        sparse_matrix = sp.csr_matrix((distances, (row_indices, col_indices)), shape=(num_samples, num_samples))
        logging.info(f"Created sparse k-NN graph with {sparse_matrix.nnz} edges.")
        return sparse_matrix
