import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.util import to_pandas
from gluonts.mx import SimpleFeedForwardEstimator, Trainer

# dataset has train,test and metadata
# train is an iterable think ListDataset : a list of timeseries
# each timeserie is a dict with {"target":the timeserie,"start":start date of the first value}

# https://ts.gluon.ai/stable/tutorials/forecasting/extended_tutorial.html


def explore_dataset():
    dataset = get_dataset("m4_hourly")
    # train, test and metadata
    entry = next(iter(dataset.train))  # you iterate on different timeseries in fact
    # In [8]: entry.keys()
    # Out[8]: dict_keys(['target', 'start', 'feat_static_cat', 'item_id'])
    #  'start': Period('1750-01-01 00:00', 'h'),
    # 'feat_static_cat': array([0], dtype=int32),
    # 'item_id': 0}
    # entry['target'][:3] Out[10]: array([605., 586., 586.], dtype=float32)

    train_series = to_pandas(entry)
    # In [17]: train_series
    # Out[17]:
    # 1750-01-01 00:00    605.0
    # 1750-01-01 01:00    586.0
    # 1750-01-01 02:00    586.0

# TODO: try to predict a cosinus structure please !!!


def old():
    ii = 9
    #plt.plot(ts_its[ii].values[:, 0])
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ts_its[ii].plot(ax=ax)
        nhorizon = forecast_its[ii].samples.shape[1]
        for ho in range(nhorizon):
            ns = pd.Series(forecast_its[ii].samples[:, ho], index=ts_its[ii].index).to_frame(f'f{ho}')
            ns.plot(ax=ax, alpha=0.5)
        #plt.plot(forecast_its[ii].samples[:, 0])
        plt.show()
    plt.plot(forecast_its[ii].mean)
    plt.plot(forecast_its[ii].quantile(0.5))

    forecast_its[ii].plot(show_label=True)
    plt.show()


# ipython -i -m crptmidfreq.season.gluonts_tuto
if __name__ == '__main__':
    N = 10  # number of time series
    T = 5_000  # number of timesteps
    prediction_length = 24
    freq = "1H"
    rd = {}
    for i in range(N):
        rd[f'noise{i}'] = np.random.normal(scale=0.2, size=T)
        offset = np.random.uniform(0, 100, size=1)
        period = np.random.uniform(20, 200, size=1)
        rd[f'pred{i}'] = np.cos(np.pi*2/period*np.arange(T))
    dfo = pd.DataFrame(rd)

    start = pd.Period("01-01-2019", freq=freq)  # can be different for each time series
    # train dataset: cut the last window of length "prediction_length", add "target" and "start" fields

    # test dataset: use the whole dataset, add "target" and "start" fields
    list_ds = []
    for i in range(N):
        list_ds += [{'target': rd[f'pred{i}']+rd[f'noise{i}'],
                     'start':start}]
    train_ds = ListDataset(list_ds, freq=freq)

    list_ds = []
    for i in range(N):
        for k in range(int(T/2), T):
            list_ds += [{'target': (rd[f'pred{i}']+rd[f'noise{i}'])[:k],
                         'start':start,
                         'item_id':f'{i}_{k}',
                         }]
    eval_ds = ListDataset(list_ds, freq=freq)

    estimator = SimpleFeedForwardEstimator(
        num_hidden_dimensions=[10],
        prediction_length=5,  # forecast_its[ii].samples.shape[1]
        context_length=100,  # forecast_its[ii].samples.shape[0]
        # context_length – Number of time steps prior to prediction time that the model takes as inputs (default: 10 * prediction_length).
        trainer=Trainer(ctx="cpu", epochs=10, learning_rate=1e-1, num_batches_per_epoch=100),
    )

    predictor = estimator.train(train_ds)
    print('Predicting now')
    pred_ds = predictor.predict(eval_ds)
    pred_its = list(pred_ds)  # pred_its[ii].samples.shape == context_lenght * prediction_length

    for i in range(N):
        rd[f'pred{i}_mean0'] = np.zeros(T)
    for pred_loc in pred_its:
        item = int(pred_loc.item_id.split('_')[0])
        num = int(pred_loc.item_id.split('_')[1])
        #rd[f'pred{item}_mean0'][num] = pred_loc.mean[-1]
        rd[f'pred{item}_mean0'][num] = pred_loc.quantile(0.5)[-1]

    ii = 8
    df_t = pd.DataFrame({
        'original': rd[f'pred{ii}'],
        'pred': rd[f'pred{ii}_mean0'],
    })
    df_t.plot(alpha=0.5)
    plt.show()
