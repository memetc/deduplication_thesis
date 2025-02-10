import numpy as np
import pandas as pd

class DataLoader:
    def __init__(self, column_name="intensities_raw"):
        self.column_name = column_name

    def load_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extracts the intensities from a DataFrame column and returns a 2D NumPy array.
        """
        data_list = df[self.column_name].tolist()
        data_array = np.array(data_list, dtype=np.float32)
        return data_array
