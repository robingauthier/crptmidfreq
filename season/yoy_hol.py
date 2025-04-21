import pandas as pd

from crptmidfreq.season.hols import generate_event_calendar_with_reverse

# only difference is that a holiday calendar is respected


def deseasonalize_yoy_hol(df,
                          date_col='date',
                          stock_col='dscode',
                          serie_col='serie',
                          caldf=None,
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
    if caldf is None:
        caldf = generate_event_calendar_with_reverse(start_year=2010, end_year=2030)

    assert operation in ['diff', 'ratio', 'lag']
    # compute "same weekday one year ago" as 52 weeks back
    if 'date_p1y_th' not in df.columns:
        df['date_p1y_th'] = df[date_col] + pd.DateOffset(weeks=52)
    if 'date_m1y_th' not in df.columns:
        df['date_m1y_th'] = df[date_col] - pd.DateOffset(weeks=52)
    if 'date_o' not in df.columns:
        df['date_o'] = df[date_col]
    if 'date_p1y_ho' not in df.columns:
        df = df.merge(caldf[['date', 'date_p1y']]
                      .rename(columns={'date_p1y': 'date_p1y_ho'}), on='date', how='left')
    if 'date_p1y' not in df.columns:
        df['date_p1y'] = df['date_p1y_ho'].fillna(df['date_p1y_th'])

    # prepare a helper table with (stock, lag_date) → lagged series value
    df_lag = (
        df[[stock_col, 'date_p1y', 'date_o', serie_col]]
        .copy()
        .rename(columns={'date_p1y': date_col, serie_col: f'{serie_col}_lag1y'})
    )

    df = df.sort_values(date_col)
    df_lag = df_lag.sort_values(date_col)

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
        rfeats += [f'{serie_col}_yoy_pct']
    elif operation == 'diff':
        ndf[f'{serie_col}_yoy'] = (ndf[serie_col] - ndf[f'{serie_col}_lag1y'])
        rfeats += [f'{serie_col}_yoy']
    elif operation == 'lag':
        rfeats += [f'{serie_col}_lag1y']
    else:
        raise(ValueError('not possible'))
    if 'is_yoy' in ndf.columns:
        ndf[f'is_yoy'] = 1*(ndf['date_m1y'] == ndf['date_o'])

    return ndf, rfeats
