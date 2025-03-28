import os
import shutil
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
# Adjust this import path to wherever your ModelStepper is defined
from crptmidfreq.stepper.incr_model import ModelStepper

# pytest ./crptmidfreq/stepper/tests/test_incr_model.py --pdb --maxfail=1


def test_model_stepper_sklearn_linear():
    """Test the ModelStepper with a scikit-learn LinearRegression model."""
    # 1) Setup a test folder
    test_folder = "test_model_stepper_linear"
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    os.makedirs(test_folder)

    # 2) Create a model generator that returns an sklearn linear model
    def model_gen():
        return LinearRegression()

    # 3) Instantiate ModelStepper
    lookback = 500
    minlookback = 200
    fitfreq = 100
    stepper = ModelStepper(
        folder=test_folder,
        name="test_modelstep_linear",
        lookback=lookback,
        minlookback=minlookback,
        fitfreq=fitfreq,
        model_gen=model_gen,
        with_fit=True,
    )

    # 4) Generate some dummy data
    np.random.seed(42)
    n_samples = 1000
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
    assert corr > 0.8


def test_model_stepper_sklearn_linear_changing_n():
    """ what happens if we add more stocks along the line?"""
    # 1) Setup a test folder
    test_folder = "test_model_stepper_linear"
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    os.makedirs(test_folder)

    # 2) Create a model generator that returns an sklearn linear model
    def model_gen():
        return LinearRegression()

    # 3) Instantiate ModelStepper
    lookback = 500
    minlookback = 200
    fitfreq = 100
    stepper = ModelStepper(
        folder=test_folder,
        name="test_modelstep_linear",
        lookback=lookback,
        minlookback=minlookback,
        fitfreq=fitfreq,
        model_gen=model_gen,
        with_fit=True,
    )

    # 4) Generate some dummy data
    np.random.seed(42)
    n_samples = 1000
    n_features = 3

    # Timestamps can be simple range or actual dates
    dts = np.arange(n_samples)
    xseries = np.random.randn(n_samples, n_features)
    yserie = 2.0 * xseries[:, 0] + 0.5 * xseries[:, 1] - xseries[:, 2] + 0.3 * np.random.randn(n_samples)
    wgtserie = 1.0 + 0.1 * np.random.rand(n_samples)  # always > 1

    xseries2 = np.random.randn(n_samples, n_features+3)

    preds = stepper.update(dts, xseries, yserie, wgtserie)
    stepper.save()

    preds2 = stepper.update(dts, xseries2, yserie, wgtserie)

    # it should just not fail...
