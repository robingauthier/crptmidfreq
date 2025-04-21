
"""
N-BEATS Model.
"""
from torch.optim import AdamW
from torch import nn
import torch
from typing import Tuple
import pandas as pd
import numpy as np
import torch as t
from skorch.callbacks import EarlyStopping
from skorch import NeuralNetRegressor
# https://github.com/ServiceNow/N-BEATS/blob/master/models/nbeats.py


class NBeatsBlock(t.nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """

    def __init__(self,
                 input_size,
                 theta_size: int,
                 basis_function: t.nn.Module,
                 layers: int,
                 layer_size: int):
        """
        N-BEATS block.

        :param input_size: Insample size.
        :param theta_size:  Number of parameters for the basis function.
        :param basis_function: Basis function which takes the parameters and produces backcast and forecast.
        :param layers: Number of layers.
        :param layer_size: Layer size.
        """
        super().__init__()
        self.layers = t.nn.ModuleList([t.nn.Linear(in_features=input_size, out_features=layer_size)] +
                                      [t.nn.Linear(in_features=layer_size, out_features=layer_size)
                                       for _ in range(layers - 1)])
        self.basis_parameters = t.nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        block_input = x
        for layer in self.layers:
            block_input = t.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)


class NBeats(t.nn.Module):
    """
    N-Beats Model.
    """

    def __init__(self, blocks: t.nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(self, x: t.Tensor) -> t.Tensor:

        residuals = x.flip(dims=(1,))
        #input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast)
            forecast = forecast + block_forecast
        return forecast


class GenericBasis(t.nn.Module):
    """
    Generic basis function.
    """

    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: t.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class TrendBasis(t.nn.Module):
    """
    Polynomial function to model trend.
    """

    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1  # degree of polynomial with constant term
        self.backcast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=np.float32) / backcast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32),
            requires_grad=False)
        self.forecast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=np.float32) / forecast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32), requires_grad=False)

    def forward(self, theta: t.Tensor):
        backcast = t.einsum('bp,pt->bt', theta[:, self.polynomial_size:], self.backcast_time)
        forecast = t.einsum('bp,pt->bt', theta[:, :self.polynomial_size], self.forecast_time)
        return backcast, forecast


class SeasonalityBasis(t.nn.Module):
    """
    Harmonic functions to model seasonality.
    """

    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.frequency = np.append(np.zeros(1, dtype=np.float32),
                                   np.arange(harmonics, harmonics / 2 * forecast_size,
                                             dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
            np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        forecast_grid = 2 * np.pi * (
            np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        self.backcast_cos_template = t.nn.Parameter(t.tensor(np.transpose(np.cos(backcast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.backcast_sin_template = t.nn.Parameter(t.tensor(np.transpose(np.sin(backcast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.forecast_cos_template = t.nn.Parameter(t.tensor(np.transpose(np.cos(forecast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.forecast_sin_template = t.nn.Parameter(t.tensor(np.transpose(np.sin(forecast_grid)), dtype=t.float32),
                                                    requires_grad=False)

    def forward(self, theta: t.Tensor):
        params_per_harmonic = theta.shape[1] // 4
        backcast_harmonics_cos = t.einsum('bp,pt->bt', theta[:, 2 * params_per_harmonic:3 * params_per_harmonic],
                                          self.backcast_cos_template)
        backcast_harmonics_sin = t.einsum('bp,pt->bt', theta[:, 3 * params_per_harmonic:], self.backcast_sin_template)
        backcast = backcast_harmonics_sin + backcast_harmonics_cos
        forecast_harmonics_cos = t.einsum('bp,pt->bt',
                                          theta[:, :params_per_harmonic], self.forecast_cos_template)
        forecast_harmonics_sin = t.einsum('bp,pt->bt', theta[:, params_per_harmonic:2 * params_per_harmonic],
                                          self.forecast_sin_template)
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast


# -----------------------------------------------------------------------------
# Factory for building your block stack
# -----------------------------------------------------------------------------

def make_nbeats_blocks(
    input_size: int,
    output_size: int,
    trend_blocks: int = 1,
    trend_layers: int = 2,
    trend_layer_size: int = 128,
    degree_of_polynomial: int = 2,
    seasonality_blocks: int = 1,
    seasonality_layers: int = 2,
    seasonality_layer_size: int = 128,
    num_of_harmonics: int = 4,
    generic_blocks: int = 0,
    generic_layers: int = 2,
    generic_layer_size: int = 128,
):
    blocks = []
    # trend stacks
    for _ in range(trend_blocks):
        tbasis = TrendBasis(
            degree_of_polynomial=degree_of_polynomial,
            backcast_size=input_size,
            forecast_size=output_size,
        )
        blocks.append(
            NBeatsBlock(
                input_size=input_size,
                theta_size=tbasis.polynomial_size * 2,
                basis_function=tbasis,
                layers=trend_layers,
                layer_size=trend_layer_size,
            )
        )
    # seasonality stacks
    for _ in range(seasonality_blocks):
        sbasis = SeasonalityBasis(
            harmonics=num_of_harmonics,
            backcast_size=input_size,
            forecast_size=output_size,
        )
        blocks.append(
            NBeatsBlock(
                input_size=input_size,
                theta_size=max(1, 4 * int(
                    np.ceil(num_of_harmonics / 2 * output_size)
                    - (num_of_harmonics - 1)
                )),
                basis_function=sbasis,
                layers=seasonality_layers,
                layer_size=seasonality_layer_size,
            )
        )
    # generic stacks
    for _ in range(generic_blocks):
        gbasis = GenericBasis(
            backcast_size=input_size,
            forecast_size=output_size,
        )
        blocks.append(
            NBeatsBlock(
                input_size=input_size,
                theta_size=input_size + output_size,
                basis_function=gbasis,
                layers=generic_layers,
                layer_size=generic_layer_size,
            )
        )

    return nn.ModuleList(blocks)


# -----------------------------------------------------------------------------
# The skorch wrapper
# -----------------------------------------------------------------------------
class NBeatsNet(NeuralNetRegressor):
    def __init__(
        self,
        input_size: int,

        output_size: int = 1,
        # stack sizes
        trend_blocks: int = 1,
        trend_layers: int = 2,
        trend_layer_size: int = 128,
        degree_of_polynomial: int = 2,
        seasonality_blocks: int = 1,
        seasonality_layers: int = 2,
        seasonality_layer_size: int = 128,
        num_of_harmonics: int = 4,
        generic_blocks: int = 0,
        generic_layers: int = 2,
        generic_layer_size: int = 128,
        # optimizer & training
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        betas: tuple = (0.9, 0.999),
        max_epochs: int = 15,
        batch_size: int = 64,
        device: str = 'cpu',
        **kwargs,
    ):
        # build the blocks once
        blocks = make_nbeats_blocks(
            input_size,
            output_size,
            trend_blocks,
            trend_layers,
            trend_layer_size,
            degree_of_polynomial,
            seasonality_blocks,
            seasonality_layers,
            seasonality_layer_size,
            num_of_harmonics,
            generic_blocks,
            generic_layers,
            generic_layer_size,
        )

        # module factory for skorch
        def module_factory():
            return NBeats(blocks)

        super().__init__(
            module=module_factory,
            criterion=nn.MSELoss,           # your loss here
            optimizer=AdamW,
            optimizer__lr=lr,
            optimizer__weight_decay=weight_decay,
            optimizer__betas=betas,
            train_split=None,               # no val‐split by default
            max_epochs=max_epochs,
            batch_size=batch_size,
            device=device,
            # callbacks=[EarlyStopping(patience=10)],
            **kwargs,
        )

    def predict(self, X):
        """Override to flatten (N,1) → (N,) and keep pandas Index if present."""
        Xindex = None
        if isinstance(X, (pd.DataFrame, pd.DataFrame)):
            Xindex = X.index
            X = X.values
        X = np.asarray(X, dtype='float32')
        preds = super().predict(X)
        # skorch returns shape (N,1) by default, so:
        preds = preds.reshape(-1)
        if Xindex is not None:
            return pd.Series(preds, index=Xindex)
        return preds

    def fit(self, X, y=None, **fit_params):
        # convert pandas to numpy
        if isinstance(X, (pd.DataFrame, pd.DataFrame)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        X = np.asarray(X, dtype='float32')
        if y is not None:
            y = np.asarray(y, dtype='float32')
        nX = t.tensor(X).to(self.device)
        ny = t.tensor(y).to(self.device)
        return super().fit(nX, ny, **fit_params)
