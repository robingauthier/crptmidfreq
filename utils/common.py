import logging

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(message)s',  # Specify the log message format
    datefmt='%Y-%m-%d::%H:%M:%S'  # Format for the timestamp
)


def set_pandas_display():
    import pandas as pd

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.expand_frame_repr', False)


def clean_folder(folder='feats_simple_v1'):
    import shutil
    import os
    from config_loc import get_data_folder
    path = os.path.join(get_data_folder(), folder)
    if os.path.exists(path):
        shutil.rmtree(path)


def to_csv(df, name):
    from config_loc import get_analysis_folder
    for iter in range(20):
        try:
            niter = '' if iter == 0 else str(iter)
            df.to_csv(get_analysis_folder() + name + f'{niter}.csv')
            break
        except Exception as e:
            pass


def print_ram_usage():
    import psutil
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1e9
    print(f'RAM usage {ram_usage}Go')
