import logging
import pandas as pd
import numpy as np
from crptmidfreq.config_loc import get_analysis_folder

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(message)s',  # Specify the log message format
    datefmt='%Y-%m-%d::%H:%M:%S'  # Format for the timestamp
)

def lvals(df,level):
    return df.index.get_level_values(level)

def filter_date_df(df,start_date=None,end_date=None):
    if isinstance(df.index,pd.MultiIndex):
        if start_date is not None:
            df=df[lvals(df,'date')>=start_date]
        if end_date is not None:
            df=df[lvals(df,'date')<=end_date]
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
    path = os.path.join(get_feature_folder(), folder)
    if os.path.exists(path):
        shutil.rmtree(path)


def to_csv(df, name):
    from crptmidfreq.config_loc import get_analysis_folder
    for iter in range(20):
        try:
            niter = '' if iter == 0 else str(iter)
            tpath=get_analysis_folder() + name + f'{niter}.csv'
            df.to_csv(tpath)
            print(f'Saved : {tpath}')
            break
        except Exception as e:
            pass

def rename_key(featd,old,new):
    featd[new] = featd.pop(old)
    return featd

def print_ram_usage():
    import psutil
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1e9
    print(f'RAM usage {ram_usage}Go')


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
    for k,v in kwargs.items():
        if not isinstance(v, np.ndarray):
            raise ValueError(f"All inputs must be numpy arrays --see: {k}")
    
    # Minimal type validation
    assert dt.dtype in ['<M8[us]','<M8[D]','<M8[m]','<M8[ns]','int64']
    assert dscode.dtype in ['int64','object']
    
    # Validate that all inputs have the same length
    if len(dt) != len(dscode):
        raise ValueError("All inputs must have the same length")
    for k,v in kwargs.items():
        if len(dt) != len(v):
            raise ValueError(f"All inputs must have the same length --see: {k}")
    