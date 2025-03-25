
from functools import partial
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib

import matplotlib.pyplot as plt
from crptmidfreq.stepper.incr_diff import DiffStepper
from .common import clean_folder
# python ./utils/bktester.py


def ann_sr(pnl, ann_factor=252.0, ddof=1):
    ret = ann_factor * np.mean(pnl)
    sigma = sqrt(ann_factor) * np.std(pnl, ddof=ddof)
    return np.nan if sigma == 0 else ret / sigma


def mdd_ret_ratio(pnl, ann_factor=252.0):
    mdd = max(dd_series(pnl))
    ret = np.mean(pnl) * ann_factor
    return np.nan if ret == 0 else mdd / ret


qtiles = [0.025, 0.05, 0.1, 0.9, 0.95, 0.975]
qtags = [f'yq_{int(100 * q)}' for q in qtiles]
latest_n_list = [1]
latest_sr_tags = [f'sr_{n}' for n in latest_n_list]
tqtiles = [0.5, 0.75, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.997, 0.999]
tq_tags = [f'tq_{100 * x:.1f}' for x in tqtiles]


def get_ann_factor(dt):
    """infers the sampling from dt
    the result is in years
    """
    median_diff_dt = np.median(np.diff(dt.view(np.int64)))
    dt_diff_years = median_diff_dt/1e6/3600/24/365
    return 1/dt_diff_years


def dd_series(pnl):
    """drawdown formula"""
    cumul = np.cumsum(pnl)
    peak = np.maximum.accumulate(cumul)
    return peak - cumul


def get_daily_stats(dt, tot_pnl, tot_trd, tot_gmv, rd=None, suf=''):
    """
    Here there is no dscode, things must be aggregated
    dt: datetime numpy array
    r : P&L numpy array
    tot_trd:trade notional
    tot_gmv : gmv
    """
    if rd is None:
        rd = {
            f'sr{suf}': np.nan,
            f'mdd{suf}': np.nan,
            f'rpt{suf}': np.nan,
            f'rog{suf}': np.nan,
            f'ann_pnl{suf}': np.nan,
            f'factor{suf}': np.nan,
            f'sigma{suf}': np.nan,
            f'srs{suf}': np.nan,  # list of srs by year
        }
    if tot_pnl.shape[0] < 10:
        return rd
    ret = np.mean(tot_pnl)
    sigma = np.std(tot_pnl, ddof=1)
    true_ann_factor = get_ann_factor(dt)
    epsilon = 1e-6
    max_dt = np.max(dt)
    rd[f'cnt{suf}'] = dt.shape[0]
    rd[f'sr{suf}'] = np.sqrt(true_ann_factor) * ret / (sigma + epsilon)
    rd[f'ann_pnl{suf}'] = ret * true_ann_factor/1e3
    rd[f'mdd{suf}'] = 100*np.max(dd_series(tot_pnl)) / tot_gmv.mean() if tot_gmv.mean() > 0 else np.nan
    rd[f'rpt{suf}'] = 1e4 * tot_pnl.sum() / tot_trd.sum() if tot_trd.sum() > 0 else np.nan
    rd[f'rog{suf}'] = 1e2 * tot_pnl.sum() / tot_gmv.mean() if tot_gmv.mean() > 0 else np.nan
    rd[f'factor{suf}'] = true_ann_factor
    # ann_pnl /= 1e3  # convert to thousands
    # sharpe in the last N years
    latest_srs = []
    for n in latest_n_list:
        year_to_ms = 365 * 24 * 60 * 60 * 1_000_000  # seconds → microseconds pd.Timedelta(days=365 * n)
        mod_r = tot_pnl[dt >= (max_dt - n*year_to_ms)]
        try:
            latest_sr = np.sqrt(true_ann_factor) * np.mean(mod_r) / np.std(mod_r, ddof=1)
        except:
            print(f'failed to calculate latest sr for prev {n=} years')
            latest_sr = np.nan
        latest_srs.append(latest_sr)
    rd[f'srs{suf}'] = latest_sr
    return rd


def get_impact_bps_vwap(ypred_trd=None, turnover=None, volatility=None, spread=None):
    alpha_coef = 0.125
    sigma_power = 1
    pov_power = 0.5
    delta_coef = 0.15

    volatility_bps = volatility / np.sqrt(252) * 1e4
    participation = np.clip(ypred_trd / turnover, 0.0, 1.0)
    impact_bps_spread = delta_coef * spread
    impact_bps = impact_bps_spread+(
        alpha_coef * np.power(volatility_bps, sigma_power) *
        np.power(participation, pov_power)
    )
    return impact_bps


def bktest_stats(
        dt,           # numpy array of np.datetime64
        dscode,       # numpy array of instrument codes (used in DiffStepper)
        ypred,        # numpy array of predicted signal values
        y,            # numpy array of forward returns
        lots,         # numpy array of weights
        name,         # column or signal name
        spread=None,  # numpy array; market spread (bps)
        turnover=None,  # numpy array; turnover values
        volatility=None,  # numpy array; volatility values
        borrow=None,      # numpy array; financing borrow cost (bps)
        long_rate=None,   # numpy array; long financing rate (bps)
        comms=0,          # commission cost (bps) (scalar or array)
        costs=True,
        max_participation=None,  # scalar cap on participation rate
        save_graph_path=None,    # file path for saving a plot
        save_dailypnl_path=None,  # file path for saving daily pnl Excel file
        out_dailypnl=False,  # to stop at dailypnl computation
):
    """
    Compute backtest statistics using numpy arrays.

    All inputs (except save paths) are numpy arrays. We use pandas only to
    aggregate daily data (by converting dt to daily dates).

    Returns a dictionary (rd) of performance metrics.
    """
    # Initialize results dictionary with default NaNs.
    rd = {
        'name': name,
        'sr': np.nan,
        'mdd': np.nan,
        'rpt': np.nan,
        'rog': np.nan,
        'ann_pnl': np.nan,
        'factor': np.nan,
        'sigma': np.nan,
        'srs': None,
        'ypred_qtiles': None,
        'trd_cost': np.nan,
        'comm_cost': np.nan,
        'fin_cost': np.nan,
        'net_sr2': np.nan,
        'avg_pos': np.nan,
        'avg_gmv': np.nan,
        'med_gmv': np.nan,
        'ypred_std': np.nan
    }

    # If ypred is all zero (after replacing nans with 0), return rd.
    ypred_clean = np.where(np.isnan(ypred), 0.0, ypred)
    if np.all(ypred_clean == 0.0):
        return rd

    # Replace inf with nan in ypred.
    ypred_clean = np.where(np.isinf(ypred_clean), np.nan, ypred_clean)

    # Compute quantiles for ypred.
    rd['ypred_std'] = np.nanstd(ypred_clean)
    rd['ypred_qtiles'] = np.nanquantile(ypred_clean, q=[0.025, 0.05, 0.1, 0.9, 0.95, 0.975])

    # Ensure lots and y are set (fill nans with defaults).
    lots = np.where(np.isnan(lots), 1.0, lots)
    y = np.where(np.isnan(y), 0.0, y)

    # Effective signal s = ypred * lots.
    s = np.nan_to_num(ypred_clean, nan=0.0) * lots

    # Cap s if max_participation is provided.
    if (max_participation is not None):
        assert turnover is not None
        s = np.where(np.abs(s) > max_participation * turnover,
                     np.sign(s) * max_participation * turnover, s)
        # Recompute ypred from the capped signal.
        ypred_clean = np.where(lots > 0, s / lots, 0.0)

    # Compute gross PnL.
    gross_pnl = s * y
    # Replace any inf values and nans.
    gross_pnl = np.where(np.isinf(gross_pnl), 0.0, gross_pnl)
    gross_pnl = np.where(np.isnan(gross_pnl), 0.0, gross_pnl)

    # Trade volume is taken as absolute signal.
    clean_folder(folder='bktester')
    m = DiffStepper(folder='bktestser', name='diff')
    trd0 = m.update(dt, dscode, s)
    trd = np.abs(trd0)

    # --------------------------
    # Transaction Costs
    # --------------------------
    if costs:
        # Commission cost (assumed comms is scalar or a numpy array of same length as y).
        comm_cost = trd * 1e-4 * comms

        # Market impact cost: if spread, turnover, and volatility are provided.
        if (spread is not None) and (turnover is not None) and (volatility is not None):
            impact_bps = get_impact_bps_vwap(ypred_trd=trd,
                                             turnover=turnover,
                                             volatility=volatility,
                                             spread=spread)
            trd_cost = trd * 1e-4 * impact_bps
        else:
            trd_cost = np.zeros_like(trd)

        # Financing cost: if borrow and long_rate are provided.
        if (borrow is not None) and (long_rate is not None):
            fin_cost = np.abs(s) * 1e-4 * np.where(s > 0, long_rate, np.abs(borrow)) * (1/252)
        else:
            fin_cost = np.zeros_like(s)

        net_pnl = gross_pnl - trd_cost - comm_cost - fin_cost
    else:
        net_pnl = gross_pnl

    # Aggregation (using pandas here)
    # Build a DataFrame for aggregation
    df_daily = pd.DataFrame({
        'dt': dt,
        'gross_pnl': gross_pnl,
        'net_pnl': net_pnl,
        'trd': trd,
        'gmv': np.abs(s),
    })
    df_daily_ag = df_daily.groupby('dt').sum().reset_index()
    daily_dt = df_daily_ag['dt'].to_numpy()
    daily_gross_pnl = df_daily_ag['gross_pnl'].to_numpy()
    daily_net_pnl = df_daily_ag['net_pnl'].to_numpy()
    daily_trd = df_daily_ag['trd'].to_numpy()
    daily_gmv = df_daily_ag['gmv'].to_numpy()

    # If too few days, return rd.
    if df_daily_ag.size < 10:
        return rd
    if out_dailypnl:
        return {
            'daily_dt': daily_dt,
            'daily_net_pnl': daily_net_pnl,
            'daily_gross_pnl': daily_gross_pnl,
            'daily_trd': daily_trd,
            'daily_gmv': daily_gmv,
        }
    tot_trd = np.sum(daily_trd)
    tot_gmv = np.sum(daily_gmv)
    rd['trd_cost'] = 1e4 * np.sum(trd_cost) / tot_trd if tot_trd > 0 else np.nan
    rd['comm_cost'] = 1e4 * np.sum(comm_cost) / tot_trd if tot_trd > 0 else np.nan
    rd['fin_cost'] = 1e4 * np.sum(fin_cost) / tot_gmv if tot_gmv > 0 else np.nan

    # syntax : get_daily_stats(dt,tot_pnl, tot_trd, tot_gmv,rd=None,suf='')
    rd = get_daily_stats(daily_dt, daily_gross_pnl, daily_trd, daily_gmv, rd=rd)
    rd = get_daily_stats(daily_dt, daily_net_pnl, daily_trd, daily_gmv, rd=rd, suf='_net')

    rd['avg_gmv'] = np.mean(daily_gmv) / 1e6   # in millions
    rd['med_gmv'] = np.median(daily_gmv) / 1e6   # in millions

    # --------------------------
    # Optional: Save Graphs and Daily PnL
    # --------------------------
    if save_graph_path is not None:
        # Plot cumulative net pnl (using daily_net from pandas aggregation)
        cum_net = np.cumsum(daily_net_pnl)
        fig, ax = plt.subplots(figsize=(10, 6))
        daily_dt_f = pd.to_datetime(daily_dt*1e3)
        ax.plot(daily_dt_f, cum_net)
        ax.set_title('Cumulative Net PnL')
        tsave_graph_path = save_graph_path.replace('NAME', name)
        plt.savefig(tsave_graph_path)
        plt.close(fig)
        print(f'Bktest Graph saved to {tsave_graph_path}')

    if save_dailypnl_path is not None:
        # Create a DataFrame to export daily pnl stats.
        epsilon = 1e-8
        df_daily.set_index('dt', inplace=True)
        df_daily.sort_index(inplace=True)
        df_daily['cumpnl'] = df_daily['pnl'].cumsum()
        df_daily['cumpnlpct'] = df_daily['cumpnl'] / (df_daily['grossdelta'].quantile(0.8) + epsilon)
        tsave_dailypnl_path = save_dailypnl_path.replace('NAME', name)
        df_daily.to_excel(tsave_dailypnl_path)
        print(f'Bktest CSV saved to {tsave_dailypnl_path}')

    return rd
