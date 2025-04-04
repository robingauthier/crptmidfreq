from filelock import FileLock
import os
import pickle
import shutil
import hashlib
import gc
from collections.abc import MutableMapping
from crptmidfreq.config_loc import get_feature_folder
from crptmidfreq.utils.common import get_logger
logger = get_logger()


def get_sha256(string):
    data = string.encode()
    hash_obj = hashlib.sha256(data)
    hex_digest = hash_obj.hexdigest()
    return hex_digest


class LazyDict(MutableMapping):
    def __init__(self, folder='lazydict', clean=False):
        self.folder = os.path.join(get_feature_folder(), folder) + '/'
        if os.path.exists(self.folder) and clean:
            shutil.rmtree(self.folder)
        os.makedirs(self.folder, exist_ok=True)
        self.debug=False

        self.lock_path = os.path.join(self.folder, "lock.lock")
        self.lock = FileLock(self.lock_path)

    def __getstate__(self):
        # Remove the lock from the object's state before pickling.
        state = self.__dict__.copy()
        if 'lock' in state:
            del state['lock']
        return state

    def __setstate__(self, state):
        # Restore instance attributes and reinitialize the lock.
        self.__dict__.update(state)
        self.lock_path = os.path.join(self.folder, "lock.lock")
        self.lock = FileLock(self.lock_path)

    def __setitem__(self, key, value):
        if self.debug:
            logger.info(f'LazyDict {self.folder} set {key}')
        with self.lock:
            #hexd = get_sha256(key)
            #new_file = f"value_{hexd}.pkl"
            new_file = f"{key}.pkl"
            new_path = os.path.join(self.folder, new_file)

            with open(new_path, "wb") as f:
                pickle.dump(value, f)
        gc.collect()

    def __getitem__(self, key):
        if self.debug:
            logger.info(f'LazyDict get {self.folder} set {key}')
        with self.lock:
            if key not in self.keys():
                raise KeyError(key)
            filename = f"{key}.pkl"
            path = os.path.join(self.folder, filename)
            if not os.path.exists(path):
                raise KeyError(f"Data file for key '{key}' is missing!")
            with open(path, "rb") as f:
                value = pickle.load(f)
        return value

    def __delitem__(self, key):
        with self.lock:
            if key not in self.keys():
                raise KeyError(key)
            filename = f"{key}.pkl"
            path = os.path.join(self.folder, filename)
            if os.path.exists(path):
                os.remove(path)

    def __len__(self):
        with self.lock:
            return len(self.keys())

    def __iter__(self):
        with self.lock:
            return iter(list(self.keys()))

    def __contains__(self, key):
        with self.lock:
            return key in self.keys()

    def keys(self):
        l = os.listdir(self.folder)
        l = [x for x in l if not x in ['index.pkl', 'lock.lock']]
        l = [x[:-4] for x in l]
        return l

    def items(self):
        with self.lock:
            for k in self.keys():
                yield (k, self[k])

    def values(self):
        with self.lock:
            for k in self.keys():
                yield self[k]

    def clear(self):
        with self.lock:
            for k, fname in self.keys().items():
                path = os.path.join(self.folder, fname)
                if os.path.exists(path):
                    os.remove(path)
            self.keys().clear()
            self._save_index()

    def __repr__(self):
        with self.lock:
            keys = list(self.keys().keys())
        return f"LazyDict(folder={self.folder!r}, keys={keys!r})"
