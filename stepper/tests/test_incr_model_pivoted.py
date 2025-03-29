import os
import shutil
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from numba.typed import Dict
from numba.core import types
# Adjust this import path to wherever your ModelStepper is defined
from crptmidfreq.stepper.incr_model_pivoted import PivotModelStepper
from crptmidfreq.stepper.incr_pivot import PivotStepper
from crptmidfreq.utils.common import merge_dicts

# pytest ./crptmidfreq/stepper/tests/test_incr_model_pivoted.py --pdb --maxfail=1


def test_pivoted_model_stepper():
    """Test the ModelStepper with a scikit-learn LinearRegression model."""
    # 1) Setup a test folder
    cfg = dict(
        kmeans_k=20,
        kmeans_lookback=10000,
        kmeans_fitfreq=100
    )

    test_folder = "test_model_stepper_kmeans"
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    os.makedirs(test_folder)

    # 4) Generate some dummy data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Timestamps can be simple range or actual dates
    dts = np.arange(n_samples)
    xseries = np.random.randn(n_samples, n_features)

    pxd = Dict.empty(
        key_type=types.int64,
        value_type=types.Array(types.float64, 1, 'C'))
    for i in range(n_features):
        pxd[i] = xseries[:, i].flatten()

    def model_gen_kmeans():
        return KMeans(n_clusters=3, random_state=42, n_init='auto')
    cls_model = PivotModelStepper \
        .load(folder=test_folder,
              name=f'model_kmeans',
              lookback=cfg.get('kmeans_lookback'),
              minlookback=500,
              fitfreq=cfg.get('kmeans_fitfreq'),
              gap=1,
              model_gen=model_gen_kmeans,
              is_kmeans=True,
              with_fit=True)
    kmeansres, ndts = cls_model.update(dts, pxd, yserie=None, wgtserie=None)

    rdf = pd.DataFrame(dict(kmeansres))
    assert rdf.shape[1] == n_features
    assert len(ndts) <= len(dts)
    print(rdf)


def test_pivoted_model_stepper_with_univ():
    """Test the ModelStepper with a scikit-learn LinearRegression model."""
    # 1) Setup a test folder
    cfg = dict(
        kmeans_k=20,
        kmeans_lookback=10000,
        kmeans_fitfreq=100
    )

    test_folder = "test_model_stepper_kmeans"
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    os.makedirs(test_folder)

    # 4) Generate some dummy data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Timestamps can be simple range or actual dates
    dts = np.arange(n_samples)
    xseries = np.random.randn(n_samples, n_features)
    univ = np.sign(np.abs(np.random.randn(n_samples, n_features)))

    pxd = Dict.empty(
        key_type=types.int64,
        value_type=types.Array(types.float64, 1, 'C'))
    for i in range(n_features):
        pxd[i] = xseries[:, i].flatten()

    pu = Dict.empty(
        key_type=types.int64,
        value_type=types.Array(types.float64, 1, 'C'))
    for i in range(n_features):
        pu[i] = univ[:, i].flatten()

    def model_gen_kmeans():
        return KMeans(n_clusters=3, random_state=42, n_init='auto')
    cls_model = PivotModelStepper \
        .load(folder=test_folder,
              name=f'model_kmeans',
              lookback=cfg.get('kmeans_lookback'),
              minlookback=500,
              fitfreq=cfg.get('kmeans_fitfreq'),
              gap=1,
              model_gen=model_gen_kmeans,
              is_kmeans=True,
              with_fit=True)
    kmeansres, ndts = cls_model.update(dts, pxd, pu)

    rdf = pd.DataFrame(dict(kmeansres))
    assert rdf.shape[1] == n_features
    assert len(ndts) <= len(dts)
    print(rdf)


def test_pivoted_model_stepper_with_univ2():
    """Test the ModelStepper with a scikit-learn LinearRegression model."""
    # 1) Setup a test folder
    cfg = dict(
        kmeans_k=20,
        kmeans_lookback=10000,
        kmeans_fitfreq=100
    )

    test_folder = "test_model_stepper_kmeans"
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    os.makedirs(test_folder)

    # 4) Generate some dummy data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Timestamps can be simple range or actual dates
    dts = np.arange(n_samples)
    xseries = np.random.randn(n_samples, n_features)
    univ = np.sign(np.abs(np.random.randn(n_samples, n_features)))

    pxd = Dict.empty(
        key_type=types.int64,
        value_type=types.Array(types.float64, 1, 'C'))
    for i in range(n_features):
        pxd[i] = xseries[:, i].flatten()

    pu = Dict.empty(
        key_type=types.int64,
        value_type=types.Array(types.float64, 1, 'C'))
    for i in range(n_features):
        pu[i] = univ[:, i].flatten()

    def model_gen_kmeans():
        return KMeans(n_clusters=3, random_state=42, n_init='auto')
    cls_model = PivotModelStepper \
        .load(folder=test_folder,
              name=f'model_kmeans',
              lookback=cfg.get('kmeans_lookback'),
              minlookback=500,
              fitfreq=cfg.get('kmeans_fitfreq'),
              gap=1,
              model_gen=model_gen_kmeans,
              is_kmeans=True,
              with_fit=True)
    kmeansres, ndts = cls_model.update(dts, pxd, pu)
    cls_model.save()
    kmeansres, ndts = cls_model.update(dts, pxd, pu)

    rdf = pd.DataFrame(dict(kmeansres))
    assert rdf.shape[1] == n_features
    assert len(ndts) <= len(dts)
    print(rdf)
