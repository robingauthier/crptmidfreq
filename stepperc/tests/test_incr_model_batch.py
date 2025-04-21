import os
import shutil

import numpy as np
import pandas as pd

from crptmidfreq.mllib.feedforward_v1 import FeedForwardNet
# Adjust this import path to wherever your ModelStepper is defined
from crptmidfreq.stepper.incr_model_batch import ModelBatchStepper

# pytest ./crptmidfreq/stepper/tests/test_incr_model_batch.py --pdb --maxfail=1


def test_model_stepper_sklearn_linear():
    """Test the ModelStepper with a scikit-learn LinearRegression model."""
    # 1) Setup a test folder
    test_folder = "test_model_stepper_linear"
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    os.makedirs(test_folder)

    # 2) Create a model generator that returns an sklearn linear model
    def model_gen(n_features=10):
        return FeedForwardNet(input_dim=n_features, hidden_dim=3, output_dim=1)

    # 3) Instantiate ModelStepper
    lookback = 5000
    minlookback = 200
    fitfreq = 1000
    stepper = ModelBatchStepper(
        folder=test_folder,
        name="test_modelstep_linear",
        lookback=lookback,
        ramlookback=1000,
        minlookback=minlookback,
        fitfreq=fitfreq,
        model_gen=model_gen,
        with_fit=True,
    )

    # 4) Generate some dummy data
    np.random.seed(42)
    n_samples = 10000
    n_features = 3

    # Timestamps can be simple range or actual dates
    dts = np.arange(n_samples)
    xseries = np.random.randn(n_samples, n_features)
    yserie = 2.0 * xseries[:, 0] + 0.5 * xseries[:, 1] - xseries[:, 2] + 0.3 * np.random.randn(n_samples)
    wgtserie = 1.0 + 0.1 * np.random.rand(n_samples)  # always > 1

    # 5) Update the stepper and get predictions
    preds = stepper.update(dts, xseries, yserie, wgtserie)

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
    assert corr > 0.6


def test_model_stepper_sklearn_linear_multi():
    """Test the ModelStepper with a scikit-learn LinearRegression model."""
    # 1) Setup a test folder
    test_folder = "test_model_stepper_linear"
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    os.makedirs(test_folder)

    # 2) Create a model generator that returns an sklearn linear model
    def model_gen(n_features=10):
        return FeedForwardNet(input_dim=n_features, hidden_dim=3, output_dim=1)

    # 3) Instantiate ModelStepper
    lookback = 5000
    minlookback = 200
    fitfreq = 1000
    stepper = ModelBatchStepper(
        folder=test_folder,
        name="test_modelstep_linear",
        lookback=lookback,
        ramlookback=1000,
        minlookback=minlookback,
        fitfreq=fitfreq,
        model_gen=model_gen,
        with_fit=True,
    )

    # 4) Generate some dummy data
    np.random.seed(42)
    n_samples = 10000
    n_features = 3

    # Timestamps can be simple range or actual dates
    dts = np.arange(n_samples)
    xseries = np.random.randn(n_samples, n_features)
    yserie = 2.0 * xseries[:, 0] + 0.5 * xseries[:, 1] - xseries[:, 2] + 0.3 * np.random.randn(n_samples)
    wgtserie = 1.0 + 0.1 * np.random.rand(n_samples)  # always > 1

    # 5) Update the stepper and get predictions
    c1 = 1000
    c2 = 5000
    c3 = 8000
    preds1 = stepper.update(dts[:c1], xseries[:c1], yserie[:c1], wgtserie[:c1])
    preds2 = stepper.update(dts[c1:c2], xseries[c1:c2], yserie[c1:c2], wgtserie[c1:c2])
    preds3 = stepper.update(dts[c2:c3], xseries[c2:c3], yserie[c2:c3], wgtserie[c2:c3])
    import pdb
    pdb.set_trace()

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
    assert corr > 0.6
