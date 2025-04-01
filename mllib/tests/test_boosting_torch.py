import numpy as np
import pandas as pd
import os
import torch
from crptmidfreq.config_loc import get_feature_folder
from crptmidfreq.mllib.iterable_data import ParquetIterableDataset
from crptmidfreq.mllib.boosting_torch import gen_boosting_torch
from crptmidfreq.mllib.train_pytorch import train_model
from crptmidfreq.utils.common import clean_folder
from crptmidfreq.mllib.tests.test_feedforward import save_some_parquet_files
from crptmidfreq.mllib.tests.test_feedforward import create_synthetic_data
np.random.seed(42)


# pytest ./crptmidfreq/mllib/tests/test_boosting_torch.py --pdb --maxfail=1

g_folder = os.path.join(get_feature_folder(), 'test_ml')+'/'


def test_train():
    # 1) Create our streaming dataset
    m = 20
    clean_folder(g_folder)
    os.makedirs(g_folder, exist_ok=True)
    save_some_parquet_files(m=m)
    model = gen_boosting_torch(m)
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
