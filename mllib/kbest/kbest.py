import numpy as np
import pandas as pd
from crptmidfreq.stepper import *
from crptmidfreq.utils.common import clean_folder
from crptmidfreq.stepper.zregistry import StepperRegistry
from crptmidfreq.utils.common import rename_key
from crptmidfreq.utils.common import get_hash
from crptmidfreq.utils.common import ewm_alpha
from crptmidfreq.utils.common import get_logger
from crptmidfreq.utils.common import get_sig_cols
from crptmidfreq.utils.common import get_sigf_cols
from crptmidfreq.utils.common import to_csv
from crptmidfreq.utils.lazy_dict import LazyDict
from joblib import Parallel, delayed
from functools import partial
g_reg = StepperRegistry()
logger = get_logger()

# another lib.py


def perform_kbest_loc(featd,
                      pnld,
                      col,
                      retcol='sret',
                      wgtcol='wgt',
                      window=1000,
                      folder=None,
                      clip_pnlpct=0.01,  # clip P&L vs gross on stock and daily level
                      name=None,
                      debug=False,
                      sharpe_th=3.0,
                      dd_th=5.0,
                      rpt_th=2.0,
                      hold_th=24*60*5,
                      r=g_reg):
    wlag = 1
    cls_lag = RollingLagStepper \
        .load(folder=f"{folder}", name=f"kbest{name}_{col}_lag{wlag}", window=wlag)
    lag_val = cls_lag.update(pnld['dtsi'], featd['dscode'], featd[col])
    r.add(cls_lag)

    # Creating P&L column
    pnld[f'{col}_s'] = np.nan_to_num(lag_val)*pnld[wgtcol]
    pnld[f'{col}_pnl'] = pnld[f'{col}_s']*pnld[retcol]

    # Computing the trading
    cls_diff = DiffStepper \
        .load(folder=f"{folder}", name=f"kbest{name}_{col}_trd", window=1)
    pnld[f'{col}_trd'] = cls_diff.update(pnld['dtsi'], pnld['dscode'], pnld[f'{col}_s'])
    r.add(cls_diff)
    pnld[f'{col}_trd_abs'] = np.abs(pnld[f'{col}_trd'])

    pnld[f'{col}_gross'] = np.abs(pnld[f'{col}_s'])
    pnld[f'{col}_pnlpct'] = np.divide(
        pnld[f'{col}_pnl'],
        pnld[f'{col}_gross'],
        out=np.zeros_like(pnld[f'{col}_gross']),
        where=~np.isclose(pnld[f'{col}_gross'], np.zeros_like(pnld[f'{col}_gross'])))

    # Correcting for outliers
    pnld[f'{col}_pnlc'] = np.where(
        np.abs(pnld[f'{col}_pnlpct']) < clip_pnlpct,
        pnld[f'{col}_pnl'],
        pnld[f'{col}_gross']*np.sign(pnld[f'{col}_pnl'])*clip_pnlpct)

    # Aggregating at the date level
    pnlgd = LazyDict(folder=folder+f'/kbest{name}{col}g')  # aggregated at dt level
    cls_gsum_gross = GroupbySumStepper\
        .load(folder=f"{folder}", name=f"kbest{name}_{col}_sum_gross")
    rts, rby, rval = cls_gsum_gross.update(pnld['dtsi'], pnld['dscode'], pnld['all'], pnld[f'{col}_gross'])
    r.add(cls_gsum_gross)
    pnlgd['dtsi'] = rts
    pnlgd[f'{col}_gross'] = rval

    cls_gsum_pnl = GroupbySumStepper\
        .load(folder=f"{folder}", name=f"kbest{name}_{col}_sum_pnl")
    rts, rby, rval = cls_gsum_pnl.update(pnld['dtsi'], pnld['dscode'], pnld['all'], pnld[f'{col}_pnl'])
    r.add(cls_gsum_pnl)
    pnlgd[f'{col}_pnl'] = rval

    cls_gsum_trd = GroupbySumStepper\
        .load(folder=f"{folder}", name=f"kbest{name}_{col}_sum_trd")
    rts, rby, rval = cls_gsum_trd.update(pnld['dtsi'], pnld['dscode'], pnld['all'], pnld[f'{col}_trd_abs'])
    r.add(cls_gsum_trd)
    pnlgd[f'{col}_trd'] = rval

    pnlgd['dscode'] = np.ones_like(pnlgd['dtsi'], dtype=np.int64)

    ######################### STATS ############
    # now computing Sharpe
    cls_ewm_pnl = EwmStepper\
        .load(folder=f"{folder}", name=f"kbest{name}_{col}_gewm{window}_pnl", window=window)
    pnlgd[f'{col}_pnl_ewm'] = cls_ewm_pnl.update(pnlgd['dtsi'], pnlgd['dscode'], pnlgd[f'{col}_pnl'])
    r.add(cls_ewm_pnl)

    cls_ewm_trd = EwmStepper\
        .load(folder=f"{folder}", name=f"kbest{name}_{col}_gewm{window}_trd", window=window)
    pnlgd[f'{col}_trd_ewm'] = cls_ewm_trd.update(pnlgd['dtsi'], pnlgd['dscode'], pnlgd[f'{col}_trd'])
    r.add(cls_ewm_trd)

    cls_ewm_gross = EwmStepper\
        .load(folder=f"{folder}", name=f"kbest{name}_{col}_gewm{window}_gross", window=window)
    pnlgd[f'{col}_gross_ewm'] = cls_ewm_gross.update(pnlgd['dtsi'], pnlgd['dscode'], pnlgd[f'{col}_gross'])
    r.add(cls_ewm_gross)

    cls_ewmstd = EwmStdStepper\
        .load(folder=f"{folder}", name=f"kbest{name}_{col}_gewmstd{window}_pnl", window=window)
    pnlgd[f'{col}_pnl_ewmstd'] = cls_ewmstd.update(pnlgd['dtsi'], pnlgd['dscode'], pnlgd[f'{col}_pnl'])
    r.add(cls_ewmstd)

    pnlgd['cnt'] = np.ones_like(pnlgd['dtsi'], dtype=np.int64)
    cls_cnt = CumSumStepper \
        .load(folder=f"{folder}", name=f"kbest{name}_{col}_gcnt")
    cpnl = cls_cnt.update(pnlgd['dtsi'], pnlgd['dscode'], pnlgd[f'cnt'])
    pnlgd['cnt'] = cpnl
    r.add(cls_cnt)

    # Drawdown
    cls_cum = CumSumStepper \
        .load(folder=f"{folder}", name=f"kbest{name}_{col}_gcumsum")
    cpnl = cls_cum.update(pnlgd['dtsi'], pnlgd['dscode'], pnlgd[f'{col}_pnl'])
    r.add(cls_cum)

    cls_emax = MaxStepper \
        .load(folder=f"{folder}", name=f"kbest{name}_{col}_csum_emax")
    cpnlmax = cls_emax.update(pnlgd['dtsi'], pnlgd['dscode'], cpnl)
    r.add(cls_emax)

    cls_emin = MinStepper \
        .load(folder=f"{folder}", name=f"kbest{name}_{col}_csum_emax")
    cpnlmin = cls_emin.update(pnlgd['dtsi'], pnlgd['dscode'], cpnl)
    r.add(cls_emax)

    pnlgd[f'{col}_cpnl'] = cpnl
    pnlgd[f'{col}_pnl_dd'] = cpnlmax-cpnl
    pnlgd[f'{col}_pnl_ddn'] = cpnl-cpnlmin

    # dividing by the avg gmv to get a percentage number
    num = pnlgd[f'{col}_pnl_dd']
    denum = pnlgd[f'{col}_gross_ewm']
    ddpct = np.divide(
        num,
        denum,
        out=np.zeros_like(denum),
        where=~np.isclose(denum,
                          np.zeros_like(denum)))
    pnlgd[f'{col}_pnl_dd'] = ddpct * 100.0

    # case we want to short the signal
    num = pnlgd[f'{col}_pnl_ddn']
    denum = pnlgd[f'{col}_gross_ewm']
    ddpct = np.divide(
        num,
        denum,
        out=np.zeros_like(denum),
        where=~np.isclose(denum,
                          np.zeros_like(denum)))
    pnlgd[f'{col}_pnl_ddn'] = ddpct * 100.0

    unit = np.median(np.diff(pnlgd['dtsi']))
    unit_day = 3600*24*1_000_000
    scaling_factor = unit_day*365/unit

    denum = pnlgd[f'{col}_pnl_ewmstd']
    pnlgd[f'{col}_pnl_sharpe'] = np.divide(
        pnlgd[f'{col}_pnl_ewm'],
        denum,
        out=np.zeros_like(denum),
        where=~np.isclose(denum, np.zeros_like(denum)))
    pnlgd[f'{col}_pnl_sharpe'] = np.sqrt(scaling_factor)*pnlgd[f'{col}_pnl_sharpe']

    denum = pnlgd[f'{col}_trd_ewm']
    pnlgd[f'{col}_pnl_rpt'] = np.divide(
        pnlgd[f'{col}_pnl_ewm'],
        denum,
        out=np.zeros_like(denum),
        where=~np.isclose(denum, np.zeros_like(denum)))
    pnlgd[f'{col}_pnl_rpt'] = np.nan_to_num(pnlgd[f'{col}_pnl_rpt']*1e4)

    # holding period = gmv/trd
    denum = pnlgd[f'{col}_trd_ewm']
    pnlgd[f'{col}_pnl_hold'] = np.divide(
        pnlgd[f'{col}_gross_ewm'],
        denum,
        out=np.zeros_like(denum),
        where=~np.isclose(denum, np.zeros_like(denum)))
    pnlgd[f'{col}_pnl_hold'] = np.nan_to_num(pnlgd[f'{col}_pnl_hold'])

    ############## SELECTION RULES #########
    # first rule has a prior which is go long this signal !

    pnlgd[f'{col}_sel'] = (1.0 *
                           (pnlgd[f'{col}_pnl_sharpe'] > sharpe_th) *
                           (pnlgd[f'{col}_pnl_dd'] < dd_th) *
                           (pnlgd[f'{col}_pnl_rpt'] > rpt_th) *
                           (pnlgd[f'{col}_pnl_hold'] < hold_th) *
                           (pnlgd[f'cnt'] > 100)
                           - 1.0 *
                           (pnlgd[f'{col}_pnl_sharpe'] < -1.0*sharpe_th) *
                           (pnlgd[f'{col}_pnl_ddn'] < dd_th) *
                           (pnlgd[f'{col}_pnl_rpt'] < -1.0*rpt_th) *
                           (pnlgd[f'{col}_pnl_hold'] < hold_th) *
                           (pnlgd[f'cnt'] > 100)
                           )

    #### MERGING BACK #######
    featd['all'] = np.ones_like(featd['dtsi'], dtype=np.int64)
    pnlgd['all'] = np.ones_like(pnlgd['dtsi'], dtype=np.int64)
    cls_merge = MergeAsofStepper \
        .load(folder=f"{folder}", name=f"kbest{name}_{col}_masof")
    merge_val = cls_merge.update(featd['dtsi'], featd['all'],
                                 pnlgd['dtsi'], pnlgd['all'], pnlgd[f'{col}_sel'])
    featd[f'kbest_{col}_sel'] = merge_val
    featd[f'kbest_{col}_xsel'] = featd[f'kbest_{col}_sel']*featd[f'{col}']
    featd[f'kbest_sum'] = featd[f'kbest_sum']+featd[f'kbest_{col}_xsel']
    featd[f'kbest_cnt'] = featd[f'kbest_cnt']+np.abs(featd[f'kbest_{col}_sel'])
    stats = {'col': col,
             'abs_sel': np.mean(np.abs(featd[f'kbest_{col}_sel'])),
             'sel': np.mean(featd[f'kbest_{col}_sel']),
             'sr': np.mean(pnlgd[f'{col}_pnl_sharpe']),
             'rpt': np.mean(pnlgd[f'{col}_pnl_rpt']),
             'dd': np.mean(pnlgd[f'{col}_pnl_dd']),
             'ddn': np.mean(pnlgd[f'{col}_pnl_ddn']),
             'hold': np.mean(pnlgd[f'{col}_pnl_hold']),
             }
    if debug:
        cls_merge_dd = MergeAsofStepper \
            .load(folder=f"{folder}", name=f"kbest{name}_{col}_masof")
        merge_val = cls_merge_dd.update(featd['dtsi'], featd['all'],
                                        pnlgd['dtsi'], pnlgd['all'], pnlgd[f'{col}_pnl_dd'])
        featd[f'kbest_{col}_dd'] = merge_val

        cls_merge_sh = MergeAsofStepper \
            .load(folder=f"{folder}", name=f"kbest{name}_{col}_masof")
        merge_val = cls_merge_sh.update(featd['dtsi'], featd['all'],
                                        pnlgd['dtsi'], pnlgd['all'], pnlgd[f'{col}_pnl_sharpe'])
        featd[f'kbest_{col}_sharpe'] = merge_val

        dfg = pd.DataFrame({k: pnlgd[k] for k in pnlgd.keys()})
        to_csv(dfg, f'debug_kbest_{col}')
    return stats


def perform_kbest(featd,
                  retcol='sret',
                  wgtcol='wgt',
                  window=1000,
                  folder=None,
                  clip_pnlpct=0.01,  # clip P&L vs gross on stock and daily level
                  name=None,
                  debug=False,
                  sharpe_th=3.0,
                  dd_th=5.0,
                  rpt_th=2.0,
                  hold_th=24*60*5,
                  n_jobs=5,
                  r=g_reg):
    """
    retcol : how we measure the P&L
    """
    logger.info('-'*20)
    logger.info('***Start of Kbest***')
    sigfs = get_sigf_cols(featd)

    # Preparing for the results
    featd[f'kbest_sum'] = np.zeros_like(featd['dtsi'], dtype=np.float64)
    featd[f'kbest_cnt'] = np.zeros_like(featd['dtsi'], dtype=np.int64)

    # we copy the relevant data
    pnld = LazyDict(folder=folder+f'/kbest{name}')
    lstats = []

    for col in sigfs:
        pnld[col] = featd[col]
    for col in ['dtsi', 'dscode', retcol, 'wgt']:
        pnld[col] = featd[col]
    pnld['all'] = np.ones_like(featd['dtsi'], dtype=np.int64)

    perform_kbest_loc2 = partial(
        perform_kbest_loc,
        featd=featd,
        pnld=pnld,
        retcol=retcol,
        wgtcol=wgtcol,
        window=window,
        folder=folder,
        clip_pnlpct=clip_pnlpct,  # clip P&L vs gross on stock and daily level
        name=name,
        debug=debug,
        sharpe_th=sharpe_th,
        dd_th=dd_th,
        rpt_th=rpt_th,
        hold_th=hold_th,
        r=r

    )
    lstats = Parallel(n_jobs=n_jobs)(delayed(perform_kbest_loc2)(col=col) for col in sigfs)

    print('Kbest stats:::')
    statsdf = pd.DataFrame(lstats)\
        .assign(abssel=lambda x: x['sel'].abs())\
        .sort_values('abssel', ascending=False)
    print(statsdf)

    num = featd[f'kbest_sum']
    denum = featd[f'kbest_cnt']
    featd[f'sig_kbest'] = np.divide(
        num,
        denum,
        out=np.zeros_like(denum),
        where=~np.isclose(denum, np.zeros_like(denum)))

    return featd
