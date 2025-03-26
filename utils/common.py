import logging
import pandas as pd
import numpy as np
from crptmidfreq.config_loc import get_analysis_folder
from crptmidfreq.utils.log import get_logger
import hashlib


logger = get_logger()


def get_hash(text):
    # Encode the string to bytes, then compute the SHA256 hash
    hash_object = hashlib.sha256(text.encode('utf-8'))
    return hash_object.hexdigest()


def lvals(df, level):
    return df.index.get_level_values(level)


def filter_date_df(df, start_date=None, end_date=None):
    if isinstance(df.index, pd.MultiIndex):
        if start_date is not None:
            df = df[lvals(df, 'date') >= start_date]
        if end_date is not None:
            df = df[lvals(df, 'date') <= end_date]
    else:
        print('filter_date_df :: Not implemented yet')
    return df


def set_pandas_display():
    import pandas as pd

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.expand_frame_repr', False)


def clean_folder(folder='feats_simple_v1'):
    """cleans the feature folder"""
    import shutil
    import os
    from crptmidfreq.config_loc import get_feature_folder
    logger.info(f'Cleaning folder {folder}')
    path = os.path.join(get_feature_folder(), folder)
    if os.path.exists(path):
        shutil.rmtree(path)


def to_csv(df, name):
    from crptmidfreq.config_loc import get_analysis_folder
    for iter in range(20):
        try:
            niter = '' if iter == 0 else str(iter)
            tpath = get_analysis_folder() + name + f'{niter}.csv'
            df.to_csv(tpath)
            print(f'Saved : {tpath}')
            break
        except Exception as e:
            pass


def rename_key(featd, old, new):
    featd[new] = featd.pop(old)
    assert new in featd.keys()
    return featd


def get_sig_cols(featd):
    """Convention :sig_ will be signals"""
    lr = []
    for k in featd.keys():
        if k.startswith('sig_'):
            lr += [k]
    return lr


def get_sigf_cols(featd):
    """Convention :sigf_ will be features ready for ML"""
    lr = []
    for k in featd.keys():
        if k.startswith('sigf_'):
            lr += [k]
    return lr


def get_forward_cols(featd):
    """Convention :sigf_ will be features ready for ML"""
    lr = []
    for k in featd.keys():
        if k.startswith('forward_'):
            lr += [k]
    return lr


def ewm_alpha(window=1.0):
    """Convert half-life to alpha"""
    assert window > 0
    return 1 - np.exp(np.log(0.5) / window)


def print_ram_usage():
    import psutil
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1e9
    print(f'RAM usage {ram_usage:.2f} Go')


def validate_input(dt, dscode, **kwargs):
    """
    Common input validation for update methods in subclasses.

    Args:
        dt: numpy array of datetime64 values
        dscode: numpy array of categorical codes

    Raises:
        ValueError: If input validation fails.
    """
    # Validate that inputs are numpy arrays
    if not isinstance(dt, np.ndarray) \
            or not isinstance(dscode, np.ndarray):
        raise ValueError("All inputs must be numpy arrays")
    for k, v in kwargs.items():
        if not isinstance(v, np.ndarray):
            raise ValueError(f"All inputs must be numpy arrays --see: {k}")

    # Minimal type validation
    assert dt.dtype in ['<M8[us]', '<M8[D]', '<M8[m]', '<M8[ns]', 'int64']
    assert dscode.dtype in ['int64', 'object']

    # Validate that all inputs have the same length
    if len(dt) != len(dscode):
        raise ValueError("All inputs must have the same length")
    for k, v in kwargs.items():
        if len(dt) != len(v):
            raise ValueError(f"All inputs must have the same length --see: {k}")



def merge_dicts(cfg,dcfg,name=''):
    for k,v in dcfg.items():
        if k not in cfg.keys():
            logger.info(f'Missing key={k} in cfg for {name} -- will use default value')
            cfg[k]=v
    return cfg


def filter_dict_to_univ(featd):
    """filter the dictionary to a specific dscode"""
    filter_numpy = (featd['univ'] > 0)
    featd2 = {k: featd[k][filter_numpy] for k in featd.keys()}
    return featd2


def filter_dict_to_dscode(featd, dscode_str='BTCUSDT'):
    """filter the dictionary to a specific dscode"""
    filter_numpy = (featd['dscode_str'] == dscode_str)
    featd2 = {k: featd[k][filter_numpy] for k in featd.keys()}
    return featd2


def filter_dict_to_dts(featd, dtsi=1):
    """filter the dictionary to a specific dscode"""
    filter_numpy = (featd['dtsi'] == dtsi)
    featd2 = {k: featd[k][filter_numpy] for k in featd.keys()}
    return featd2



def save_features(featd, name=''):
    wcols = (get_sigf_cols(featd) +
             get_forward_cols(featd) +
             ['dtsi', 'dscode', 'close', 'wgt', 'kmeans_cat', 'univ'])
    df = pd.DataFrame({k: featd[k] for k in wcols})
    df.to_parquet(os.path.join(get_analysis_folder(), f'{name}.pq'))


def save_signal(featd, name=''):
    wcols = (get_sig_cols(featd) +
             get_forward_cols(featd) +
             ['dtsi', 'dscode', 'close', 'wgt', 'univ'])
    df = pd.DataFrame({k: featd[k] for k in wcols})
    df.to_parquet(os.path.join(get_analysis_folder(), f'{name}.pq'))
