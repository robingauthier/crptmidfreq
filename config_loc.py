import pandas as pd
pd.set_option('display.max_rows', 500)

def get_data_folder():
    # raw exchange data
    return '/Volumes/data/crpt/' # NAS

def get_data_db_folder():
    # DuckDB is not advised on NAS
    return '/Users/sachadrevet/data_tmp/'

def get_feature_folder():
    return '/Users/sachadrevet/data_tmp/features/'

def get_analysis_folder():
    return '/Users/sachadrevet/data_tmp/analysis/'
