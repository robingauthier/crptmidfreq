import torch.nn as nn
from torchensemble import GradientBoostingRegressor

from crptmidfreq.mllib.feedforward_v1 import FeedForwardNet

# https://ensemble-pytorch.readthedocs.io/en/latest/quick_start.html#choose-the-ensemble


def gen_boosting_torch(n_features=10, n_estimators=10, hidden_dim=64):
    model = GradientBoostingRegressor(
        estimator=FeedForwardNet,
        estimator_args={'input_dim': n_features, 'hidden_dim': hidden_dim},
        n_estimators=n_estimators,
        shrinkage_rate=1.0,
        cuda=False,  # default is True here
    )

    criterion = nn.MSELoss()
    model.set_criterion(criterion)
    model.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)

    return model
