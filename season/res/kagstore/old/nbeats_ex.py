import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crptmidfreq.mllib.lgbm_sklearn import LGBMModel
from crptmidfreq.mllib.lgbm_sklearn_optuna import LGBMModelOptuna
from crptmidfreq.mllib.timeclf import TimeSplitClf
from crptmidfreq.season.res.kagstore.kagstore_data import get_data
from crptmidfreq.season.res.kagstore.kagstore_feature import add_features
from crptmidfreq.season.res.kagstore.kagstore_feature import add_features_lag
from crptmidfreq.utils.common import to_csv

#from crptmidfreq.season.gluonts_v1 import fit_gluonts_model
# sales_log= log(1+sales)
# exp(sales_log)= 1+sales
# sales= exp(sales_log)-1

data_start = pd.to_datetime('2013-01-01')
valid_start = pd.to_datetime('2015-09-01')  # for optuna
test_start = pd.to_datetime('2016-09-01')
data_end = pd.to_datetime('2017-08-15')


def main_nbeats2_optuna():
    from crptmidfreq.mllib.nbeats2_sklearn_optuna import NBeatsNetOptuna
    from crptmidfreq.mllib.nbeats2_sklearn import NBeatsNet
    df, f1 = get_data()
    df = df[df['family'] == 'LIQUOR,WINE,BEER'].copy().reset_index(drop=True)
    df, f2 = add_features_lag(df, f1, use_log=False, n_lags=400)
    df_train = df.loc[lambda x: x['date'] <= test_start].copy()
    feats = f2['numerical']+f2['categorical']
    model = NBeatsNet(
        # model = NBeatsNetOptuna(
        input_size=len(feats),
        lr=1e-5,
    )
    model.fit(df_train[feats].fillna(0.0), df_train['sales'].fillna(0.0))

    # print(model)
    #ee = model.get_params()['module']()
    #total_params = sum(p.numel() for p in ee.parameters() if p.requires_grad)
    #print(f"Total trainable params: {total_params}")

    df['pred_sales'] = model.predict(df[feats].fillna(0.0))

    to_csv(df, 'nbeats2_optuna')
    import pdb
    pdb.set_trace()
    df_test = df.loc[lambda x:x['date'] > test_start]
    mae = 0.0

    return {'mae': mae, 'name': 'nbeats_optuna'}


# ipython -i -m crptmidfreq.season.res.kagstore.nbeats_ex
if __name__ == '__main__':
    main_nbeats2_optuna()
