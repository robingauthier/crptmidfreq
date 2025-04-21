import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit

from crptmidfreq.config_loc import get_analysis_folder
from crptmidfreq.utils.common import lvals

#from crptmidfreq.outlier.hampel_cython import hampel as hampel_cython

g_folder = os.path.join(get_analysis_folder(), 'hampel/')+'/'


@njit(cache=True)
def hampel_numba(data: np.ndarray, window_size: int, n_sigma: float):
    """
    Applies the Hampel filter to a 1D numpy array for outlier detection using Numba.

    Parameters:
        data (np.ndarray): 1D input array.
        window_size (int): size of the sliding window.
        n_sigma (float): number of scaled MADs to use as threshold.

    Returns:
        filtered_data (np.ndarray): copy of `data` with outliers replaced by local median.
        outlier_indices (np.ndarray): indices of detected outliers.
        medians (np.ndarray): local median at each position.
        mads (np.ndarray): local median absolute deviation at each position.
        thresholds (np.ndarray): threshold used at each position.
    """
    n = data.shape[0]
    half_w = window_size // 2

    filtered = data.copy()
    medians = np.empty(n, dtype=data.dtype)
    mads = np.empty(n, dtype=data.dtype)
    thresholds = np.empty(n, dtype=data.dtype)
    out_idx = np.empty(n, dtype=np.int64)
    count = 0

    # temporary window buffer
    win = np.empty(window_size, dtype=data.dtype)

    for i in range(n):
        # edges: not enough data to form a full window
        if i < half_w or i > n - half_w - 1:
            medians[i] = np.nan
            mads[i] = np.nan
            thresholds[i] = np.nan
            continue

        # copy window
        for j in range(window_size):
            win[j] = data[i - half_w + j]

        # compute median
        sorted_win = np.sort(win)
        if window_size & 1:
            med = sorted_win[window_size // 2]
        else:
            med = (sorted_win[window_size//2 - 1] + sorted_win[window_size//2]) * 0.5
        medians[i] = med

        # compute MAD
        for j in range(window_size):
            win[j] = abs(win[j] - med)
        sorted_dev = np.sort(win)
        if window_size & 1:
            mad = sorted_dev[window_size // 2]
        else:
            mad = (sorted_dev[window_size//2 - 1] + sorted_dev[window_size//2]) * 0.5
        mads[i] = mad

        # threshold and outlier test
        thr = n_sigma * 1.4826 * mad
        thresholds[i] = thr
        if abs(data[i] - med) > thr:
            filtered[i] = med
            out_idx[count] = i
            count += 1

    # trim indices array
    return (
        filtered,
        out_idx[:count],
        medians,
        mads,
        thresholds
    )


def serie_autocorr_fix(serie, work_on_rank=False):
    serie = serie.ffill()
    if work_on_rank:
        serie = serie.rank()
    if serie.std() == 0:
        return 1.0
    try:
        return serie.autocorr(lag=1)
    except FloatingPointError:
        return 1.0


def convert_name(name):
    return name\
        .replace(' ', '_')\
        .replace('/', '_')\
        .replace(':', '_')\
        .replace('(', '_')\
        .replace(')', '_')\
        .replace(',', '_')


def plot_outliers(df, col, dscode, window=15, folderloc=g_folder, name=''):
    ncol = convert_name(col)

    outlier_ids = df[f'{col}_outlier_id'].dropna().unique().tolist()
    outlier_ids = [x for x in outlier_ids if x > 0]

    # we will loop on each outlier and plot it
    for outlier_id in outlier_ids:
        ref_date = lvals(df[df[f'{col}_outlier_id'] == outlier_id], 'date')[0]
        ref_date_str = ref_date.strftime('%Y%m%d')

        print(
            f'Hampel filter : outlier found on {dscode} {col} - date= {ref_date}-  open '
            f'figure in checks/hampel/')
        # print(out_df)
        original_data = df[col]
        medians = df[f'{col}_median']
        thresholds = df[f'{col}_thresholds']
        fig = plt.figure(figsize=(8, 6))
        axes = fig.add_subplot()
        # Plot the original data with estimated standard deviations in the first subplot
        axes.plot(original_data, label='Original Data', color='b')
        axes.fill_between(original_data.index, medians + thresholds,
                          medians - thresholds, color='gray', alpha=0.5,
                          label='Median +- Threshold')
        axes.set_xlabel('date')
        axes.set_ylabel('value')
        axes.set_title(f'Hampel {name} {dscode} {ncol} - {ref_date_str}')
        axes.plot(ref_date, original_data.loc[ref_date], 'ro', markersize=5)  # Mark as red
        axes.legend()
        if not os.path.exists(folderloc):
            os.makedirs(folderloc, exist_ok=True)
        plt.tight_layout()
        figname = folderloc+f'/hampel_{name}_{ncol}_{dscode}_{ref_date_str}.png'
        fig.savefig(figname)
        plt.close()


def hampel_ts_loc(df,
                  cols,
                  zth=10.0,
                  window=40,
                  name='',
                  mincnt=100,
                  folderloc=g_folder,
                  with_plot=False,
                  ):
    # hampel filter to flag real issues -- we look forward here
    assert isinstance(df.index, pd.MultiIndex)
    dscode = lvals(df, 'dscode')[0]
    lr = []
    for col in cols:
        nb_out = 0
        ts = df[col].ffill().dropna()

        if ts.shape[0] <= mincnt:
            lr += [{'col': col, 'msg': 'too few samples', 'nb_out': 0}]
            continue
        pctzero = (ts == 0.0).mean()
        if pctzero > 0.9:
            lr += [{'col': col, 'msg': f'too many zero : {pctzero}', 'nb_out': 0}]
            continue

        # we don't want outliers to influence our autocorrelation measure
        semilog_ts = np.log(1 + ts.abs()) * np.sign(ts)
        autoc = serie_autocorr_fix(semilog_ts, work_on_rank=True)
        if abs(autoc) < 0.7:
            lr += [{'col': col, 'msg': f'too low autocorr {autoc}', 'nb_out': 0}]
            continue

        cntvals = len(ts.unique())
        cntvalspct = cntvals / ts.shape[0]
        if (cntvals <= 10) or (cntvalspct < 0.02):
            lr += [{'col': col, 'msg': 'categorical col', 'nb_out': 0}]
            continue

        hres = hampel_numba(ts.astype(np.float32).values, window, zth)
        outlier_indices = hres[1]
        medians = hres[2]
        median_absolute_deviations = hres[3]
        thresholds = hres[4]
        if len(outlier_indices) == 0:
            continue
        is_outlier = np.zeros(ts.shape[0])
        is_outlier[outlier_indices] = 1.0
        ndf = pd.DataFrame({col: ts.values})
        ndf['date'] = lvals(ts, 'date')
        ndf[f'{col}_median'] = medians
        ndf[f'{col}_mad'] = median_absolute_deviations
        ndf[f'{col}_thresholds'] = thresholds
        ndf[f'{col}_is_outlier'] = is_outlier
        ndf[f'{col}_outlier_id'] = np.cumsum(is_outlier)
        ndf[f'{col}_idx'] = 1
        ndf[f'{col}_idx'] = ndf[f'{col}_idx'].cumsum()
        ndf = ndf.set_index('date')

        nb_out = len(outlier_indices)
        if nb_out == 0:
            lr += [{'col': col, 'msg': 'fine', 'nb_out': 0}]
            continue
        for outlier_id in outlier_indices:
            date = df.index[outlier_id][1]
            lr += [{'col': col, 'nb_out': 1, 'msg': 'outlier', 'date': date, 'id': outlier_id}]
        if with_plot:
            plot_outliers(ndf, col, dscode, window=window, folderloc=folderloc, name=name)
    if len(lr) == 0:
        return pd.DataFrame({'col': [], 'msg': [], 'nb_out': []})
    outdf = pd.DataFrame(lr)
    outdf['nb_out'] = outdf['nb_out'].fillna(0)
    return outdf


def hampel_ts(df, cols=None,
              zth=10.0,
              window=40,
              name='',
              mincnt=100,
              folderloc=g_folder,
              with_plot=True,
              nstocks=None):
    """
    This thing looks forward and should not return anything !
    """
    lr = []
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ['dscode', 'date']
    if cols is None:
        cols = df.columns.tolist()
        cols = [col for col in cols if col not in ['dscode', 'date']]
    df = df.sort_index(level='date')
    istock = 0
    for dscode, dfloc in df.groupby('dscode'):
        istock += 1
        if nstocks is not None:
            if istock > nstocks:
                break
        rdfloc = hampel_ts_loc(dfloc,
                               cols=cols,
                               window=window,
                               zth=zth,
                               name=name,
                               mincnt=mincnt,
                               folderloc=folderloc,
                               with_plot=with_plot)
        rdfloc['dscode'] = dscode
        lr += [rdfloc]
    return pd.concat(lr, axis=0, sort=False)
