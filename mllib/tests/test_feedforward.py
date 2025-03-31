import numpy as np
import pandas as pd
import os
import torch
from crptmidfreq.config_loc import get_feature_folder
from crptmidfreq.mllib.iterable_data import ParquetIterableDataset
from crptmidfreq.mllib.feedforward_v1 import FeedForwardNet
from crptmidfreq.mllib.train_pytorch import train_model
from crptmidfreq.utils.common import clean_folder
np.random.seed(42)


# pytest ./crptmidfreq/mllib/tests/test_feedforward.py --pdb --maxfail=1

g_folder = os.path.join(get_feature_folder(), 'test_ml')+'/'


def create_synthetic_data(n=10000, m=20, k=3, noise_std=1.0):
    """
    Creates a dataset of shape (n, m) plus a target column.
    Of the m features, the first k are predictive (summed up + noise),
    and the remaining m-k are random distractors.
    """
    X = np.random.normal(loc=0.0, scale=1.0, size=(n, m))

    # "True" signal is sum of first k features
    true_signal = X[:, :k].sum(axis=1)

    # Add some noise
    y = true_signal + np.random.normal(0.0, noise_std, size=n)

    # Return the data with shape (n, m) plus y
    df = pd.DataFrame(X)
    header = [f"feature_{j}" for j in range(X.shape[1])]
    df.columns = header
    df['forward_fh1'] = y
    df['forward_fh2'] = true_signal
    return df


def save_some_parquet_files(P=10, m=20):
    """
    Splits arrays X, y into P parts and writes each part to a CSV file:
    row format: f1, f2, ..., fm, label
    """

    for i in range(P):
        df = create_synthetic_data(m=m)
        filename = f"data_{i}.pq"
        df.to_parquet(g_folder+filename)


def test_train():
    # 1) Create our streaming dataset
    m = 20
    clean_folder(g_folder)
    os.makedirs(g_folder, exist_ok=True)
    save_some_parquet_files(m=m)
    model = FeedForwardNet(input_dim=m,
                           hidden_dim=3,
                           output_dim=1)
    model = train_model(g_folder, model,
                        batch_size=50,
                        batch_up=10,
                        target='forward_fh1')
    dftest = create_synthetic_data(m=m)
    xtest = dftest[[x for x in dftest.columns if not x.startswith('forward')]].values
    xtest_tensor = torch.tensor(xtest, dtype=torch.float32)
    ypred = model.forward(xtest_tensor)
    dftest['ypred'] = ypred.detach().numpy().flatten()
    correlation = dftest[['ypred', 'forward_fh2']].corr().iloc[0, 1]
    assert correlation > 0.8
