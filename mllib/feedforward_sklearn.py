from skorch import NeuralNetRegressor
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from crptmidfreq.utils.log import get_logger

log = get_logger()


# Overall skorch does most of the heavy lifting but weight is not available
# hence in loss() we would need to add weight in
# /Users/sachadrevet/anaconda3/lib/python3.11/site-packages/skorch/net.py


class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_units=(100,), nonlin=nn.GELU):
        super().__init__()
        layers = []
        dims = [input_dim, *hidden_units, 1]
        for in_f, out_f in zip(dims, dims[1:]):
            layers.append(nn.Linear(in_f, out_f))
            layers.append(nonlin())
        layers.pop()  # remove last nonlin
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X).squeeze(-1)


class FeedForwardRegressor(NeuralNetRegressor):
    def __init__(
        self,
        input_dim,
        hidden_units=(50, 20),
        lr=1e-3,
        weight_decay=1e-5,
        betas=(0.9, 0.999),
        max_epochs=20,
        batch_size=64,
        **kwargs,           # catch any extra skorch args
    ):
        super().__init__(
            module=FeedForward,
            module__input_dim=input_dim,
            module__hidden_units=hidden_units,
            optimizer=torch.optim.AdamW,
            optimizer__lr=lr,
            optimizer__weight_decay=weight_decay,
            optimizer__betas=betas,
            max_epochs=max_epochs,
            batch_size=batch_size,
            device='cpu',
            **kwargs,
        )

    def fit(self, X, y=None, **fit_params):
        # convert pandas to numpy
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        X = np.asarray(X, dtype='float32')
        if y is not None:
            y = np.asarray(y, dtype='float32')
        nX = torch.tensor(X).to(self.device)
        ny = torch.tensor(y).to(self.device)
        return super().fit(nX, ny, **fit_params)

    def predict(self, X):
        Xindex = X.index
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        X = np.asarray(X, dtype='float32')
        nX = torch.tensor(X).to(self.device)
        res = super().predict(nX)
        res = pd.Series(res, index=Xindex)
        return res
