import os

import lightgbm as lgb  # Super important otherwise crashes python
import numpy as np

from crptmidfreq.config_loc import get_feature_folder
from crptmidfreq.mllib.nbeats2_sklearn import NBeatsNet
from crptmidfreq.mllib.tests.test_feedforward import (create_synthetic_data,
                                                      save_some_parquet_files)
from crptmidfreq.utils.common import clean_folder

np.random.seed(42)

# pytest ./crptmidfreq/mllib/tests/test_nbeats2_sklearn.py --pdb --maxfail=1

g_folder = os.path.join(get_feature_folder(), 'test_ml')+'/'


def keep_import():
    model = lgb.DaskLGBMClassifier()


def test_train():
    # 1) Create our streaming dataset
    m = 20
    clean_folder(g_folder)
    os.makedirs(g_folder, exist_ok=True)
    save_some_parquet_files(m=m)

    df = create_synthetic_data(m=m, n=100_000)
    df = df.drop(['dtsi'], axis=1)  # SUPER IMPORTANT TO REMOVE IT
    n = df.shape[0]
    dftrain = df.iloc[:n//2].copy()
    dftest = df.iloc[n//2:].copy()

    numfeats = [x for x in dftest.columns if not x.startswith('forward')]
    model = NBeatsNet(
        input_size=len(numfeats),
        trend_layer_size=20,
        seasonality_layer_size=20,
        generic_layer_size=20,
        generic_blocks=2,
    )

    model.fit(dftrain[numfeats], dftrain['forward_fh2'])
    print(model)
    ee = model.get_params()['module']()
    total_params = sum(p.numel() for p in ee.parameters() if p.requires_grad)
    print(f"Total trainable params: {total_params}")

    # TrendBlock will operate on a batch_size x len(numfeats)
    ypred = model.predict(dftest[numfeats])

    assert np.std(ypred) > 1e-14
    dftest['ypred'] = ypred
    correlation = dftest[['ypred', 'forward_fh2']].corr().iloc[0, 1]
    assert correlation > 0.5
