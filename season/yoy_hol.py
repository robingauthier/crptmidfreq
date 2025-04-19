import pandas as pd
from pandas.tseries.offsets import DateOffset
from dateutil.easter import easter

def get_thanksgiving(year):
    """Return US Thanksgiving date (4th Thursday in November) for a given year."""
    nov1 = pd.Timestamp(year=year, month=11, day=1)
    return nov1 + pd.offsets.WeekOfMonth(week=3, weekday=3)

_holiday_fns = {
    'easter': lambda y: pd.Timestamp(easter(y)),
    'thanksgiving': lambda y: get_thanksgiving(y),
}

def _parse_holiday_entry(entry):
    """
    Parse strings like 'easter+2' or 'thanksgiving' → (fn, offset_days:int)
    """
    if '+' in entry:
        name, offs = entry.split('+', 1)
        offs = int(offs)
    else:
        name, offs = entry, 0
    try:
        fn = _holiday_fns[name]
    except KeyError:
        raise ValueError(f"Unknown holiday '{name}'")
    return fn, offs

def deseasonalize_yoy(df,
                      date_col='date',
                      stock_col='dscode',
                      serie_col='serie',
                      holidays=None):
    """
    YoY deseasonalization with special handling for configured moving holidays.
    
    Parameters
    ----------
    df : pd.DataFrame
    date_col : str
    stock_col : str
    serie_col : str
    holidays : list of str, e.g. ['easter','easter+1','thanksgiving','thanksgiving+1']
    
    Returns
    -------
    pd.DataFrame, with new column '{serie_col}_yoy_pct'
    """
    if holidays is None:
        holidays = ['easter', 'thanksgiving']
    # parse into list of (fn, offset_days)
    holiday_rules = [_parse_holiday_entry(h) for h in holidays]

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['year'] = df[date_col].dt.year

    def _compute_lag(dt):
        y = dt.year
        # check each holiday rule
        for fn, offs in holiday_rules:
            hol = fn(y) + DateOffset(days=offs)
            if dt == hol:
                return fn(y-1) + DateOffset(days=offs)
        # fallback → same weekday 52 weeks ago
        return dt - DateOffset(weeks=52)

    # vectorize lag computation
    df['lag_date'] = df[date_col].map(_compute_lag)

    # build lag lookup
    df_lag = (df[[stock_col, 'lag_date', serie_col]]
              .rename(columns={'lag_date': date_col, serie_col: 'serie_lag'}))

    # merge on (stock, date) → brings in last‑year’s value where available
    out = pd.merge(df, df_lag, on=[stock_col, date_col], how='left')

    # compute % residual
    out[f'{serie_col}_yoy_pct'] = ((out[serie_col] - out['serie_lag'])
                                   / out['serie_lag']) * 100

    return out.drop(columns=['year', 'lag_date'])
