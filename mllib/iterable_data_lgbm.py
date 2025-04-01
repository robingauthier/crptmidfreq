import os
import pandas as pd
import numpy as np
import lightgbm
from torch.utils.data import IterableDataset
from crptmidfreq.config_loc import get_feature_folder


class ParquetIterableDataset(IterableDataset):
    """
    An IterableDataset that streams data from all Parquet files in a given folder.
    Assumes each row is something like: feature_1, feature_2, ..., label
    and does not load the entire folder's data in memory at once.
    """

    def __init__(self, folder_path, target='forward_fh1', filterfile=None):
        """
        Args:
            folder_path (str): Path to the folder containing Parquet files.
            num_features (int): Number of feature columns (label is assumed to be right after).
        """
        super().__init__()
        self.folder_path = os.path.join(get_feature_folder(), folder_path)+'/'
        os.makedirs(self.folder_path, exist_ok=True)
        self.num_features = -1
        self.target = target
        self.filterfile = filterfile

    def __iter__(self):
        """
        Iterates over all .parquet files in the specified folder, 
        reading them one at a time. For each file, it yields one row at a time.
        """
        # List all files in the folder, filter by .parquet
        # Sort if you need a deterministic order
        file_list = sorted([
            f for f in os.listdir(self.folder_path)
            if f.lower().endswith(".pq")
        ])
        if self.filterfile is not None:
            file_list = [x for x in file_list if self.filterfile(x)]
        else:
            print('Warning :: ParquetIterableDataset filterfile is missing')

        for filename in file_list:
            file_path = os.path.join(self.folder_path, filename)

            df = pd.read_parquet(file_path)
            if self.num_features < 0:
                self.num_features = df.shape[1]
            else:
                assert df.shape[1] == self.num_features
            assert self.target in df.columns
            cols = np.array(df.columns.tolist())
            idx_target = np.array(cols == self.target)
            idx_feats = np.array([not x.startswith('forward_') for x in cols])

            x = df[cols[idx_feats]]
            y = df[self.target]
            yield x, y
