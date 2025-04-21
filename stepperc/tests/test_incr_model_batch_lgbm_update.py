import os
import shutil

import numpy as np
import pandas as pd

from crptmidfreq.mllib.train_lgbm import gen_lgbm_lin_params
# Adjust this import path to wherever your ModelStepper is defined
from crptmidfreq.stepper.incr_model_batch import ModelBatchStepper

# pytest ./crptmidfreq/stepper/tests/test_incr_model_batch_lgbm_update.py --pdb --maxfail=1


def test_model_stepper_lgbm_update():

    # 1) Setup a test folder
    test_folder = "test_model_stepper_linear"
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    os.makedirs(test_folder)

    # 3) Instantiate ModelStepper
    lookback = 300
    minlookback = 200
    fitfreq = 100
    stepper = ModelBatchStepper(
        folder=test_folder,
        name="test_modelstep_linear",
        lookback=lookback,
        ramlookback=100,
        epochs=1,
        minlookback=minlookback,
        fitfreq=fitfreq,
        model_gen=gen_lgbm_lin_params,
        with_fit=True,
    )

    # 4) Generate some dummy data
    np.random.seed(42)
    n_samples = 10000
    c = 8000
    n_features = 3

    # Timestamps can be simple range or actual dates
    dts = np.arange(n_samples)//10
    xseries = np.random.randn(n_samples, n_features)
    yserie = 2.0 * xseries[:, 0] + 0.5 * xseries[:, 1] - xseries[:, 2] + 0.3 * np.random.randn(n_samples)
    wgtserie = 1.0 + 0.1 * np.random.rand(n_samples)  # always > 1

    # 5) Update the stepper and get predictions
    preds1 = stepper.update(dts[:c], xseries[:c], yserie[:c], wgtserie[:c])
    preds2 = stepper.update(dts[c:], xseries[c:], yserie[c:], wgtserie[c:])
    preds = np.concatenate([preds1, preds2])

    # 6) Check basic correctness
    # The shape of preds must match (n_samples,)
    assert preds.shape == (n_samples,), f"Expected preds of shape {(n_samples,)}, got {preds.shape}"

    # Ensure at least one model was fitted
    assert len(stepper.hmodels) > 0, "No models were fitted, expected at least one in hmodels."

    df = pd.DataFrame({
        'y': yserie,
        'ypred': preds,
        'wgt': wgtserie
    })
    corr = df[['y', 'ypred']].corr().iloc[0, 1]
    assert corr > 0.5
    pctzero = (df['ypred'].abs() < 1e-14).mean()
    assert pctzero < 0.25

    # check that the last 10 values are not 0
    assert not np.all(df['ypred'].iloc[-10:] == 0.0)
