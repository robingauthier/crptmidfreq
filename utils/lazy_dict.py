import os
import pickle
import uuid
import shutil
import gc
from collections.abc import MutableMapping
from crptmidfreq.config_loc import get_feature_folder


class LazyDict(MutableMapping):
    """
    A 'dictionary-like' class that saves values to disk immediately after setting,
    and frees the in-memory copy. Keys are stored in a small in-memory index for 
    iteration and file lookup. Each value is pickled to a separate file.

    Caveats:
      - Not concurrency-safe (multiple processes writing at once may conflict).
      - The entire key index is in memory (if you have millions of keys, 
        you might outgrow memory).
      - In a real system, you'd use a database or 'shelve' or something more robust.
    """

    def __init__(self, folder='lazydict', clean=False):
        """
        folder: directory path to store data files
        """
        self.folder = os.path.join(get_feature_folder(), folder)+'/'
        if os.path.exists(self.folder) and clean:
            shutil.rmtree(self.folder)
        os.makedirs(self.folder, exist_ok=True)

        # Load or create the key->filename index
        self.index_path = os.path.join(self.folder, "index.pkl")
        if os.path.exists(self.index_path):
            with open(self.index_path, "rb") as f:
                self._index = pickle.load(f)
        else:
            self._index = {}  # { key: filename }

    def _save_index(self):
        # Helper: write the key->filename dict to disk
        with open(self.index_path, "wb") as f:
            pickle.dump(self._index, f)

    def __setitem__(self, key, value):
        """
        Store value on disk in a new file. Overwrite the old file if the key exists.
        """
        # If key already exists, remove old file to avoid orphaned data
        if key in self._index:
            old_file = self._index[key]
            old_path = os.path.join(self.folder, old_file)
            if os.path.exists(old_path):
                os.remove(old_path)

        # Generate a unique filename for the new data
        new_file = f"value_{uuid.uuid4().hex}.pkl"
        new_path = os.path.join(self.folder, new_file)

        # Pickle the value to disk
        with open(new_path, "wb") as f:
            pickle.dump(value, f)

        # Update in-memory index
        self._index[key] = new_file
        self._save_index()

        # "Free" the value from memory by not storing it
        # (the function ends, so 'value' is gone from the stack scope)
        gc.collect()                       # force garbage collection

    def __getitem__(self, key):
        """
        Load the value from disk on demand.
        """
        if key not in self._index:
            raise KeyError(key)
        filename = self._index[key]
        path = os.path.join(self.folder, filename)
        if not os.path.exists(path):
            raise KeyError(f"Data file for key '{key}' is missing!")
        with open(path, "rb") as f:
            value = pickle.load(f)
        return value

    def __delitem__(self, key):
        """
        Delete the value file from disk and remove the key from the index.
        """
        if key not in self._index:
            raise KeyError(key)
        filename = self._index[key]
        path = os.path.join(self.folder, filename)
        if os.path.exists(path):
            os.remove(path)
        del self._index[key]
        self._save_index()

    def __len__(self):
        return len(self._index)

    def __iter__(self):
        """
        Iterate over keys.
        """
        return iter(self._index)

    def __contains__(self, key):
        return key in self._index

    def keys(self):
        return self._index.keys()

    def items(self):
        """
        Iterate (key, value) pairs. 
        WARNING: This will unpickle each value from disk as you iterate.
        """
        for k in self._index:
            yield (k, self[k])

    def values(self):
        """
        Iterate over values (loads from disk each time).
        """
        for k in self._index:
            yield self[k]

    def clear(self):
        """
        Remove all items from the lazy dict (and disk).
        """
        for k, fname in self._index.items():
            path = os.path.join(self.folder, fname)
            if os.path.exists(path):
                os.remove(path)
        self._index.clear()
        self._save_index()

    def __repr__(self):
        return f"LazyDict(folder={self.folder!r}, keys={list(self._index.keys())!r})"
