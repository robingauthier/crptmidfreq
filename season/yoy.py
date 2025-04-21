import pandas as pd


def deseasonalize_yoy(df,
                      date_col='date',
                      stock_col='dscode',
                      serie_col='serie',
                      operation='diff'):
    """
    For each row in df, find the value of `serie_col` exactly 52 weeks before
    (same day-of-week), and compute the percent residual:

        residual_pct = (current - lag) / lag * 100

    Assumes:
      - df[date_col] can be cast to datetime
      - df may have multiple stocks; lag is matched per stock
      - if no lagged value is found, residual_pct will be NaN
    """
    assert operation in ['diff', 'ratio', 'lag']
    # compute "same weekday one year ago" as 52 weeks back
    if 'date_p1y' not in df.columns:
        df['date_p1y'] = df[date_col] + pd.DateOffset(weeks=52)
    if 'date_m1y' not in df.columns:
        df['date_m1y'] = df[date_col] - pd.DateOffset(weeks=52)
    if 'date_o' not in df.columns:
        df['date_o'] = df[date_col]

    # prepare a helper table with (stock, lag_date) → lagged series value
    df_lag = (
        df[[stock_col, 'date_p1y', 'date_o', serie_col]]
        .copy()
        .rename(columns={'date_p1y': date_col, serie_col: f'{serie_col}_lag1y'})
    )

    # merge back on stock & date to bring in the lagged value
    if not f'{serie_col}_lag1y' in df.columns:
        ndf = pd.merge_asof(df.drop(['date_o'], axis=1), df_lag,
                            on=date_col,
                            by=stock_col,
                            direction='backward')
    else:
        ndf = df
    # compute percent residual
    rfeats = []
    if operation == 'ratio':
        ndf[f'{serie_col}_yoy_pct'] = (
            (ndf[serie_col] - ndf[f'{serie_col}_lag1y'])
            / (ndf[serie_col] + ndf[f'{serie_col}_lag1y'])*2.0
        )
        ndf[f'{serie_col}_yoy_pct'] = ndf[f'{serie_col}_yoy_pct'].clip(lower=-0.5, upper=0.5)
        rfeats += [f'{serie_col}_yoy_pct']
    elif operation == 'diff':
        ndf[f'{serie_col}_yoy'] = (ndf[serie_col] - ndf[f'{serie_col}_lag1y'])
        ndf[f'{serie_col}_yoy'] = ndf[f'{serie_col}_yoy'].clip(lower=-0.5, upper=0.5)
        rfeats += [f'{serie_col}_yoy']
    elif operation == 'lag':
        rfeats += [f'{serie_col}_lag1y']
    else:
        raise(ValueError('not possible'))
    if 'is_yoy' in ndf.columns:
        ndf[f'is_yoy'] = 1*(ndf['date_m1y'] == ndf['date_o'])

    return ndf, rfeats
