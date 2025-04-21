import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset

from crptmidfreq.utils.log import get_logger

log = get_logger()


def fit_gluonts_model(df,
                      fd,
                      estimator,
                      dscode_col='dscode',
                      date_col='dtsi',
                      target_col='sales',
                      name='gluonts',
                      ):
    min_dt = df['dtsi'].min()
    max_dt = df['dtsi'].max()
    nbdays = (max_dt-min_dt)

    assert df.index.duplicated().sum() == 0
    df = df.sort_values(date_col)

    train_end = min_dt+nbdays*0.7
    train_df = df[df['dtsi'] <= train_end]
    assert train_df.shape[0] > 0.5*df.shape[0]

    list_ds = []
    for dscode, dfg in train_df.groupby(dscode_col):
        serie = dfg.set_index(date_col)[target_col]
        start = pd.to_datetime('2020-01-01')
        list_ds += [{'target': serie,
                     'start': start,
                     'item_id': dscode, }]
    train_ds = ListDataset(list_ds, freq='D')

    log.info('Training now')
    predictor = estimator.train(train_ds)

    lparams = predictor.network.collect_params()
    print(lparams)
    nparams = int(np.sum([np.prod(p.shape) for _, p in lparams.items()]))
    print(f'We have a total number of {nparams} parameters')

    log.info('Predicting now')
    pred_ds = predictor.predict(train_ds)
    temp1 = next(iter(pred_ds))
    horizon = temp1.samples.shape[1]
    context = temp1.samples.shape[0]
    #assert context >= 10

    for i in range(horizon):
        df[f'{name}_predh{i}_mean'] = np.zeros(df.shape[0])

    for dscode, dfg in df.groupby(dscode_col):
        serie = dfg.set_index(date_col)[target_col]
        start = pd.to_datetime('2020-01-01')
        dtsis = dfg[date_col].values
        nloc = dfg.shape[0]
        if np.all(dtsis <= train_end):
            continue
        nstart = np.argmax(dtsis > train_end)
        group_idx = dfg.index.to_numpy()
        for k in range(nstart, nloc):
            list_ds = [{'target': serie.iloc[:k],
                        'start': start,
                        'item_id': dscode}]
            eval_ds = ListDataset(list_ds, freq='D')
            forecast_it = predictor.predict(eval_ds)
            forecast_its = list(forecast_it)
            forecast = forecast_its[0].mean  # an array of size horizon
            row_idx = group_idx[k]
            # now write each horizon‐step into the matching column
            for h, val in enumerate(forecast):
                df.at[row_idx, f"{name}_predh{h}_mean"] = val
    return df, {'test_start': train_end}
