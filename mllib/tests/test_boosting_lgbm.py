import numpy as np
import pandas as pd
import os
import lightgbm as lgb  # Super important otherwise crashes python
from crptmidfreq.config_loc import get_feature_folder
from crptmidfreq.mllib.iterable_data import ParquetIterableDataset
from crptmidfreq.mllib.lgbm_lin_v1 import gen_lgbm_lin_v1
from crptmidfreq.mllib.train_lgbm import train_model
from crptmidfreq.mllib.train_lgbm import gen_lgbm_lin_params
from crptmidfreq.utils.common import clean_folder
from crptmidfreq.mllib.tests.test_feedforward import save_some_parquet_files
from crptmidfreq.mllib.tests.test_feedforward import create_synthetic_data
np.random.seed(42)

# pytest ./crptmidfreq/mllib/tests/test_boosting_lgbm.py --pdb --maxfail=1

g_folder = os.path.join(get_feature_folder(), 'test_ml')+'/'


def keep_import():
    model = lgb.DaskLGBMClassifier()


def test_train():
    # 1) Create our streaming dataset
    m = 20
    clean_folder(g_folder)
    os.makedirs(g_folder, exist_ok=True)
    save_some_parquet_files(m=m)

    model = train_model(g_folder,
                        model_param_generator=gen_lgbm_lin_params,
                        batch_size=10000,
                        target='forward_fh1')
    dftest = create_synthetic_data(m=m)
    xtest = dftest[[x for x in dftest.columns if not x.startswith('forward')]].values
    ypred = model.predict(xtest)
    assert np.std(ypred) > 1e-14
    dftest['ypred'] = ypred
    correlation = dftest[['ypred', 'forward_fh2']].corr().iloc[0, 1]
    assert correlation > 0.8
