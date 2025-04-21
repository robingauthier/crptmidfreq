
import os
import shutil

import numpy as np
import pandas as pd

# Adjust this import path to wherever your ModelStepper is defined
from crptmidfreq.stepper.incr_model_timeclf import TimeClfStepper

# pytest ./crptmidfreq/stepper/tests/test_incr_model_timeclf_cut.py --pdb --maxfail=1


def test_model_stepper_lgbm_cut():

    # 1) Setup a test folder
    test_folder = "test_model_stepper_linear"
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    os.makedirs(test_folder)

    # 3) Instantiate ModelStepper
    np.random.seed(42)
    n_samples = 10_000  # 1000 days
    c = 1_000
    n_features = 3

    lookback = 300  # in days
    minlookback = 150
    fitfreq = 200  # in days
    stepper = TimeClfStepper(
        folder=test_folder,
        name="test_modelstep_linear",
        lookback=lookback,
        minlookback=minlookback,
        fitfreq=fitfreq,
    )

    # 4) Generate some dummy data

    # Timestamps can be simple range or actual dates
    dts = np.arange(n_samples)//10
    start_dt = pd.to_datetime('2024-01-01').value/1e3
    dts = start_dt+dts*3600*24*1e6
    xseries = np.random.randn(n_samples, n_features)
    yserie = 2.0 * xseries[:, 0] + 0.5 * xseries[:, 1] - xseries[:, 2] + 0.3 * np.random.randn(n_samples)
    wgtserie = 1.0 + 0.1 * np.random.rand(n_samples)  # always > 1

    i = 0
    jcnt = 0
    while i+c < n_samples:
        print(i)
        ei = min(i+c, n_samples)
        # 5) Update the stepper and get predictions
        stepper.update(dts[i:ei], xseries[i:ei], yserie[i:ei])
        if jcnt == 0:
            assert len(stepper.ltimes) == 0
        if jcnt >= 1:
            assert len(stepper.ltimes) > 0
        i += c
        jcnt += 1
