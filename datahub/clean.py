import shutil
import os
from config_loc import get_data_folder


# python datahub/clean.py --date_str 2025-03-11
if __name__ == '__main__':
    shutil.rmtree(get_data_folder())
    os.makedirs(get_data_folder())
    