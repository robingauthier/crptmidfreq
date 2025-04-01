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

g_reg = StepperRegistry()
logger = get_logger()


def keep_import():
    clean_folder()


def check_cols(featd, wcols):
    for col in wcols:
        if not col in featd.keys():
            print(f'Missing col {col} in featd')
            assert False


def perform_save(r=g_reg):
    """
    must be called before shutting down the ipython
    so that on restart we can continue from where we were
    """
    r.save()


def perform_ewm(featd, feats=[], windows=[1], folder=None, name=None, r=g_reg):
    """
    featd is the feature dictionary where we put our result
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        for hl in windows:
            cls_ewm = EwmStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_ewm{hl}", window=hl)
            ewm_val = cls_ewm.update(featd['dtsi'], featd['dscode'], featd[col])
            r.add(cls_ewm)
            featd[f'{col}_ewm{hl}'] = ewm_val
            nfeats += [f'{col}_ewm{hl}']
    return featd, nfeats


def perform_macd(featd, feats=[], windows=[[12, 26]], folder=None, name=None, r=g_reg):
    """
    windows=[[12,26]] means ewm_12(X) - ewm_26(X)

    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert isinstance(windows, list)
    assert isinstance(windows[0], list)
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        for hls in windows:
            assert len(hls) == 2
            assert hls[0] < hls[1]
            hlfast = hls[0]
            hlslow = hls[1]

            cls_ewm_fast = EwmStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_ewm{hlfast}_macd", window=hlfast)
            ewm_val_fast = cls_ewm_fast.update(featd['dtsi'], featd['dscode'], featd[col])
            r.add(cls_ewm_fast)

            cls_ewm_slow = EwmStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_ewm{hlslow}_macd", window=hlslow)
            ewm_val_slow = cls_ewm_slow.update(featd['dtsi'], featd['dscode'], featd[col])
            r.add(cls_ewm_slow)

            featd[f'{col}_macd{hlfast}x{hlslow}'] = ewm_val_fast-ewm_val_slow
            nfeats += [f'{col}_macd{hlfast}x{hlslow}']
    return featd, nfeats


def perform_macd_ratio(featd, feats=[], windows=[[12, 26]], folder=None, name=None, r=g_reg):
    """
    windows=[[12,26]] means ewm_12(X) / ewm_26(X)-1.0
    useful for turnover, or things like that

    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert isinstance(windows, list)
    assert isinstance(windows[0], list)
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        for hls in windows:
            assert len(hls) == 2
            assert hls[0] < hls[1]
            hlfast = hls[0]
            hlslow = hls[1]
            cls_ewm_fast = EwmStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_ewm{hlfast}", window=hlfast)
            ewm_val_fast = cls_ewm_fast.update(featd['dtsi'], featd['dscode'], featd[col])
            r.add(cls_ewm_fast)
            cls_ewm_slow = EwmStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_ewm{hlslow}", window=hlslow)
            ewm_val_slow = cls_ewm_slow.update(featd['dtsi'], featd['dscode'], featd[col])
            r.add(cls_ewm_slow)
            featd[f'{col}_macdr{hlfast}x{hlslow}'] = np.divide(
                ewm_val_fast,
                ewm_val_slow,
                out=np.ones_like(ewm_val_slow),
                where=~np.isclose(ewm_val_slow,
                                  np.zeros_like(ewm_val_slow)))
            featd[f'{col}_macdr{hlfast}x{hlslow}'] = featd[f'{col}_macdr{hlfast}x{hlslow}'] - 1.0
            nfeats += [f'{col}_macdr{hlfast}x{hlslow}']
    return featd, nfeats


def perform_macd_signal(featd, feats=[], windows=[1], folder=None, name=None, r=g_reg):
    """
    macd - ewm(macd)
    Hence:
    windows=[[12,26,9]]  
    step1 = ewm_12(X) - ewm_26(X)
    step2 = step1 - ewm_9(step1)
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert isinstance(windows, list)
    assert isinstance(windows[0], list)
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        for hls in windows:
            assert len(hls) == 3
            assert hls[0] < hls[1]
            hlfast = hls[0]
            hlslow = hls[1]
            hlsig = hls[2]

            cls_ewm_fast = EwmStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_ewm{hlfast}", window=hlfast)
            ewm_val_fast = cls_ewm_fast.update(featd['dtsi'], featd['dscode'], featd[col])
            r.add(cls_ewm_fast)

            cls_ewm_slow = EwmStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_ewm{hlslow}", window=hlslow)
            ewm_val_slow = cls_ewm_slow.update(featd['dtsi'], featd['dscode'], featd[col])
            r.add(cls_ewm_slow)

            macd = ewm_val_fast-ewm_val_slow

            cls_ewm_sig = EwmStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_macd_ewm{hlslow}", window=hlslow)
            ewm_val_sig = cls_ewm_sig.update(featd['dtsi'], featd['dscode'], macd)
            r.add(cls_ewm_sig)

            featd[f'{col}_macd{hlfast}x{hlslow}x{hlsig}'] = macd - ewm_val_sig
            nfeats += [f'{col}_macd{hlfast}x{hlslow}x{hlsig}']
    return featd, nfeats


def perform_divide(featd, numcols=[], denumcols=[], folder=None, name=None, r=g_reg):
    for col in numcols:
        assert col in featd.keys()
    for col in denumcols:
        assert col in featd.keys()
    nfeats = []
    for numcol in numcols:
        for denumcol in denumcols:
            featd[f'{numcol}div{denumcol}'] = np.divide(
                featd[numcol],
                featd[denumcol],
                out=np.zeros_like(featd[denumcol]),
                where=~np.isclose(featd[denumcol],
                                  np.zeros_like(featd[denumcol])))
            nfeats += [f'{numcol}div{denumcol}']
    return featd, nfeats


def perform_to_sig(featd, feats=[], folder=None, name=None, r=g_reg):
    nfeats = []
    for col in feats:
        featd = rename_key(featd, col, f'sig_{col}')
        nfeats += ['sig_'+col]
    return featd, nfeats


def perform_to_sigf(featd, feats=[], folder=None, name=None, r=g_reg):
    nfeats = []
    for col in feats:
        featd = rename_key(featd, col, f'sigf_{col}')
        nfeats += ['sigf_'+col]
    return featd, nfeats


def perform_divide_m1(featd, numcols=[], denumcols=[], folder=None, name=None, r=g_reg):
    """removes one once we divided"""
    for col in numcols:
        assert col in featd.keys()
    for col in denumcols:
        assert col in featd.keys()
    nfeats = []
    for numcol in numcols:
        for denumcol in denumcols:
            featd[f'{numcol}divm1{denumcol}'] = np.divide(
                featd[numcol],
                featd[denumcol],
                out=np.ones_like(featd[denumcol]),
                where=~np.isclose(featd[denumcol],
                                  np.zeros_like(featd[denumcol])))
            featd[f'{numcol}divm1{denumcol}'] = featd[f'{numcol}divm1{denumcol}']-1.0
            nfeats += [f'{numcol}div{denumcol}']
    return featd, nfeats


def perform_ewm_std(featd, feats=[], windows=[1], folder=None, name=None, r=g_reg):
    """
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        for hl in windows:
            cls_ewmstd = EwmStdStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_ewmstd{hl}", window=hl)
            ewmstd_val = cls_ewmstd.update(featd['dtsi'], featd['dscode'], featd[col])
            r.add(cls_ewmstd)
            featd[f'{col}_ewmstd{hl}'] = ewmstd_val
            nfeats += [f'{col}_ewmstd{hl}']
    return featd, nfeats


def perform_scaling_ewm(featd, feats=[], windows=[100], clip=3, folder=None, name=None, r=g_reg):
    """

    step1 = X / ewmstd(X) 
    step2 = clip(step1)

    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        for hl in windows:
            cls_ewmstd = EwmStdStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_scaled_ewmstd{hl}", window=hl)
            ewmstd_val = cls_ewmstd.update(featd['dtsi'], featd['dscode'], featd[col])
            r.add(cls_ewmstd)
            num = featd[col]
            denum = ewmstd_val
            featd[f'{col}_scaled_ewmstd{hl}'] = np.divide(
                num,
                denum,
                out=np.zeros_like(denum),
                where=~np.isclose(denum, np.zeros_like(denum)))
            featd[f'{col}_scaled_ewmstd{hl}'] = np.clip(featd[f'{col}_scaled_ewmstd{hl}'], a_min=-clip, a_max=clip)
            nfeats += [f'{col}_scaled_ewmstd{hl}']
    return featd, nfeats


def perform_detrend_ewm(featd, feats=[], windows=[1], folder=None, name=None, r=g_reg):
    """
    X - ewm(X)
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        for hl in windows:
            cls_ewm = EwmStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_detrend_ewm{hl}", window=hl)
            ewm_val = cls_ewm.update(featd['dtsi'], featd['dscode'], featd[col])
            r.add(cls_ewm)
            featd[f'{col}_detrend_ewm{hl}'] = featd[col] - ewm_val
            nfeats += [f'{col}_detrend_ewm{hl}']
    return featd, nfeats


def perform_detrend_ewm_ratio(featd, feats=[], windows=[1], folder=None, name=None, r=g_reg):
    """
    computes P/ewm(P)-1
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        for hl in windows:
            cls_ewm = EwmStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_detrendratio_ewm{hl}", window=hl)
            ewm_val = cls_ewm.update(featd['dtsi'], featd['dscode'], featd[col])
            r.add(cls_ewm)
            num = featd[col]
            denum = ewm_val
            featd[f'{col}_detrendratio_ewm{hl}'] = np.divide(
                num,
                denum,
                out=np.ones_like(denum),
                where=~np.isclose(denum, np.zeros_like(denum))
            )
            # make it be around 0.0
            featd[f'{col}_detrendratio_ewm{hl}'] = featd[f'{col}_detrendratio_ewm{hl}']-1.0
            nfeats += [f'{col}_detrendratio_ewm{hl}']
    return featd, nfeats


def perform_ewm_scaled(featd, feats=[], windows=[1], folder=None, name=None, r=g_reg):
    """ 
    step1= ewm(X) * scaling_factor / ewmstd(X) 
    step2 = clip(step1)

    ewm(X) has Var = Var(x_i)* (1-alpha)/(1+alpha)
    this is why we need to adjust for a scaling factor 
    """
    nfeats = []
    for col in feats:
        for hl in windows:
            cls_ewm = EwmStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_ewm{hl}_ewmsc", window=hl)
            ewm_val = cls_ewm.update(featd['dtsi'], featd['dscode'], featd[col])
            r.add(cls_ewm)
            alpha = 1-ewm_alpha(hl)
            num = ewm_val*np.sqrt((1+alpha)/(1-alpha))

            cls_ewmstd = EwmStdStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_ewmstd{hl}_ewmsc", window=hl)
            ewm_val2 = cls_ewmstd.update(featd['dtsi'], featd['dscode'], featd[col])
            r.add(cls_ewmstd)

            rr = np.divide(
                num,
                ewm_val2,
                out=np.zeros_like(ewm_val2),
                where=~np.isclose(ewm_val2, np.zeros_like(ewm_val2)))
            rr = np.clip(rr, a_min=-3.0, a_max=3.0)
            featd[f'{col}_ewmscaled{hl}'] = rr
            nfeats += [f'{col}_ewmscaled{hl}']
    return featd, nfeats


def perform_sharpe_ewm(featd, sigcol='', pnlcol='', window=100, folder=None, name=None, r=g_reg):
    """
    timeserie sharpe per dscode . We adjust for gmv changes.
    very similar to the above (perform_ewm_scaled)
    a sharpe of 1 annual is equivalent to a sharpe of 1/15.6 = 0.06 on daily
    """

    absserie = np.abs(featd[sigcol])
    pnlpct = np.divide(
        featd[pnlcol],
        absserie,
        out=np.zeros_like(absserie),
        where=~np.isclose(absserie, np.zeros_like(absserie)))

    cls_ewm = EwmStepper \
        .load(folder=f"{folder}", name=f"{name}_{pnlcol}_ewm{window}", window=window)
    ewm_val = cls_ewm.update(featd['dtsi'], featd['dscode'], pnlpct)
    r.add(cls_ewm)
    alpha = 1-ewm_alpha(window)
    num = ewm_val*np.sqrt((1+alpha)/(1-alpha))

    cls_ewmstd = EwmStdStepper \
        .load(folder=f"{folder}", name=f"{name}_{pnlcol}_ewmstd{window}", window=window)
    ewm_val2 = cls_ewmstd.update(featd['dtsi'], featd['dscode'], pnlpct)
    r.add(cls_ewmstd)

    rr = np.divide(
        num,
        ewm_val2,
        out=np.zeros_like(ewm_val2),
        where=~np.isclose(ewm_val2, np.zeros_like(ewm_val2)))
    rr = np.clip(rr, a_min=-3.0, a_max=3.0)
    featd[f'{sigcol}_ewmsharpe{window}'] = rr
    nfeats = [f'{sigcol}_ewmsharpe{window}']
    return featd, nfeats


def perform_ewm_skew(featd, feats=[], windows=[1], folder=None, name=None, r=g_reg):
    """
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        for hl in windows:
            cls_skew = EwmSkewStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_ewmskew{hl}", window=hl)
            skew_val = cls_skew.update(featd['dtsi'], featd['dscode'], featd[col])
            r.add(cls_skew)
            featd[f'{col}_ewmskew{hl}'] = skew_val
            nfeats += [f'{col}_ewmskew{hl}']
    return featd, nfeats


def perform_ewm_kurt(featd, feats=[], windows=[1], folder=None, name=None, r=g_reg):
    """
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        for hl in windows:
            cls_kurt = EwmKurtStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_ewmkurt{hl}", window=hl)
            kurt_val = cls_kurt.update(featd['dtsi'], featd['dscode'], featd[col])
            r.add(cls_kurt)
            featd[f'{col}_ewmkurt{hl}'] = kurt_val
            nfeats += [f'{col}_ewmkurt{hl}']
    return featd, nfeats


def perform_ffill(featd, feats=[], folder=None, name=None, r=g_reg):
    """
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_ffill = FfillStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_ffill")
        ffill_val = cls_ffill.update(featd['dtsi'], featd['dscode'], featd[col])
        r.add(cls_ffill)
        featd[f'{col}_ffill'] = ffill_val
        nfeats += [f'{col}_ffill']
    return featd, nfeats


def perform_groupby_last(featd, feats=[], folder=None, name=None, r=g_reg):
    """
    This will return you a new feature dictionary
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    nfeatd = {}
    for col in feats:
        cls_ffill = GroupbyLastStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_last")
        rts, rcode, rval = cls_ffill.update(featd['dtsi'], featd['dscode'], featd[col])
        r.add(cls_ffill)
        if 'dtsi' not in nfeatd.keys():
            nfeatd['dtsi'] = rts
            nfeatd['dscode'] = rcode
        nfeatd[col] = rval
        nfeats += [col]
    return nfeatd, nfeats


def perform_cumsum(featd, feats=[], folder=None, name=None, r=g_reg):
    """
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_cum = CumSumStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_cumsum")
        cum_val = cls_cum.update(featd['dtsi'], featd['dscode'], featd[col])
        r.add(cls_cum)
        featd[f'{col}_cumsum'] = cum_val
        nfeats += [f'{col}_cumsum']
    return featd, nfeats


def perform_cnt_exists(featd, feats=[], folder=None, name=None, r=g_reg):
    """
    cnt=1
    cumcnt=cumsum(cnt)
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    featd['one'] = np.ones(featd['dtsi'].shape)
    cls_cum = CumSumStepper \
        .load(folder=f"{folder}", name=f"{name}_cnt_exists")
    cum_val = cls_cum.update(featd['dtsi'], featd['dscode'], featd['one'])
    r.add(cls_cum)
    featd[f'cnt_exists'] = cum_val
    nfeats += [f'cnt_exists']
    return featd, nfeats


def perform_diff(featd, feats=[], windows=[1], folder=None, name=None, r=g_reg):
    """
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    assert np.array(windows).dtype == 'int64'
    nfeats = []
    for col in feats:
        for hl in windows:
            cls_diff = DiffStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_diff{hl}", window=hl)
            diff_val = cls_diff.update(featd['dtsi'], featd['dscode'], featd[col])
            r.add(cls_diff)
            featd[f'{col}_diff{hl}'] = np.nan_to_num(diff_val)
            nfeats += [f'{col}_diff{hl}']
    return featd, nfeats


def perform_lag(featd, feats=[], windows=[1], folder=None, name=None, r=g_reg):
    """
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    assert np.array(windows).dtype == 'int64'
    nfeats = []
    for col in feats:
        for hl in windows:
            if hl == 0:
                nfeats += [f'{col}']
            else:
                cls_lag = RollingLagStepper \
                    .load(folder=f"{folder}", name=f"{name}_{col}_lag{hl}", window=hl)
                lag_val = cls_lag.update(featd['dtsi'], featd['dscode'], featd[col])
                r.add(cls_lag)
                featd[f'{col}_lag{hl}'] = lag_val
                nfeats += [f'{col}_lag{hl}']
    return featd, nfeats


def perform_lag_forward(featd, feats=[], windows=[1], folder=None, name=None, r=g_reg):
    """
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    assert np.array(windows).dtype == 'int64'
    nfeats = []
    for col in feats:
        for hl in windows:
            assert hl < 0
            df = pd.DataFrame({
                'i': np.arange(len(featd['dtsi'])),
                'dts': featd['dtsi'],
                'dscode': featd['dscode'],
                'feat': featd[col],
            })
            df['feat_forward'] = df.groupby('dscode')['feat'].transform(lambda x: x.shift(hl))
            assert df['i'].is_monotonic_increasing
            featd[f'forward_{col}_lag{hl}'] = df['feat_forward'].values
            nfeats += [f'forward_{col}_lag{hl}']
    return featd, nfeats


def perform_log(featd, feats=[], folder=None, name=None, r=g_reg):
    """
    log(1+x)
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        featd[f'{col}_log1'] = np.log(1+featd[col])
        nfeats += [f'{col}_log1']
    return featd, nfeats


def perform_abs(featd, feats=[], folder=None, name=None, r=g_reg):
    """
    abs(x)
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        featd[f'{col}_abs'] = np.abs(featd[col])
        nfeats += [f'{col}_abs']
    return featd, nfeats


def perform_rolling_max(featd, feats=[], windows=[1], folder=None, name=None):
    """
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    assert np.array(windows).dtype == 'int64'
    nfeats = []
    for col in feats:
        for hl in windows:
            cls_max = RollingMaxStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_max{hl}", window=hl)
            max_val = cls_max.update(featd['dtsi'], featd['dscode'], featd[col])
            featd[f'{col}_max{hl}'] = max_val
            nfeats += [f'{col}_max{hl}']
    return featd, nfeats


def perform_rolling_min(featd, feats=[], windows=[1], folder=None, name=None):
    """
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    assert np.array(windows).dtype == 'int64'
    nfeats = []
    for col in feats:
        for hl in windows:
            cls_min = RollingMinStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_max{hl}", window=hl)
            min_val = cls_min.update(featd['dtsi'], featd['dscode'], featd[col])
            featd[f'{col}_min{hl}'] = min_val
            nfeats += [f'{col}_min{hl}']
    return featd, nfeats


def perform_expanding_min(featd, feats=[], folder=None, name=None):
    """
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_min = MinStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_expmin")
        min_val = cls_min.update(featd['dtsi'], featd['dscode'], featd[col])
        featd[f'{col}_expmin'] = min_val
        nfeats += [f'{col}_expmin']
    return featd, nfeats


def perform_expanding_max(featd, feats=[], folder=None, name=None):
    """
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_min = MaxStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_expmax")
        min_val = cls_min.update(featd['dtsi'], featd['dscode'], featd[col])
        featd[f'{col}_expmax'] = min_val
        nfeats += [f'{col}_expmax']
    return featd, nfeats


def perform_sma(featd, feats=[], windows=[1], folder=None, name=None, r=g_reg):
    """
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    assert np.array(windows).dtype == 'int64'
    nfeats = []
    for col in feats:
        for hl in windows:
            cls_sma = RollingMeanStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_sma{hl}", window=hl)
            sma_val = cls_sma.update(featd['dtsi'], featd['dscode'], featd[col])
            r.add(cls_sma)
            featd[f'{col}_sma{hl}'] = sma_val
            nfeats += [f'{col}_sma{hl}']
    return featd, nfeats


def perform_sto(featd, feats=[], windows=[1], lags=[0], folder=None, name=None):
    """
    x - min(x) / max(x)-min(x)
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    assert windows.dtype == 'int64'
    featd, lagfeats = perform_lag(featd, feats, windows=lags, folder=folder, name=name)
    nfeats = []
    for col in lagfeats:
        for hl in windows:
            cls_max = RollingMaxStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_max{hl}", window=hl)
            cls_min = RollingMinStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_min{hl}", window=hl)
            max_val = cls_max.update(featd['dtsi'], featd['dscode'], featd[col])
            min_val = cls_min.update(featd['dtsi'], featd['dscode'], featd[col])
            num = featd[col] - min_val
            denum = max_val - min_val
            featd[f'{col}_sto{hl}'] = np.divide(
                num,
                denum,
                out=np.zeros_like(denum),
                where=~np.isclose(denum, np.zeros_like(denum))
            )
            nfeats += [f'{col}_sto{hl}']
    return featd, nfeats


def perform_merge_asof(featd_l, featd_r, feats=[], key='dscode', folder=None, name=None, r=g_reg):
    """
    """
    assert 'dtsi' in featd_l.keys()
    assert key in featd_l.keys()
    assert np.all(np.diff(featd_l['dtsi']) >= 0)

    assert 'dtsi' in featd_r.keys()
    assert key in featd_r.keys()
    assert np.all(np.diff(featd_r['dtsi']) >= 0)

    if len(feats) == 0:
        feats = list(featd_r.keys())
        feats = [x for x in feats if x != 'dtsi']
        feats = [x for x in feats if x != 'dscode']

    nfeats = []
    for col in feats:
        cls_merge = MergeAsofStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_masof")
        merge_val = cls_merge.update(featd_l['dtsi'], featd_l[key],
                                     featd_r['dtsi'], featd_r[key], featd_r[col])

        r.add(cls_merge)
        featd_l[f'{col}_masof'] = merge_val
        nfeats += [f'{col}_masof']
    return featd_l, nfeats


def perform_clip(featd, feats=[], folder=None, name=None, r=g_reg, low_clip=np.nan, high_clip=np.nan):
    """
    the alpha is variable , it depends on the ufeats
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_clip = ClipStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_clip", low_clip=low_clip, high_clip=high_clip)
        featd[f'{col}_clip'] = cls_clip.update(featd['dtsi'], featd['dscode'], featd[col])
        r.add(cls_clip)
        nfeats += [f'{col}_clip']
    return featd, nfeats


def perform_quantile_global(featd, feats=[], qs=[], folder=None, name=None, r=g_reg):
    """
    we use an expanding quantile computation
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert len(qs) > 0
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_qtl = QuantileStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_qtl", qs=qs)
        arr_qtls = cls_qtl.update(featd['dtsi'], np.zeros(featd['dtsi'].shape[0], dtype=np.int64), featd[col])
        r.add(cls_qtl)
        for i in range(len(qs)):
            qs_loc = qs[i]
            featd[f'{col}_qtl{qs_loc:.2f}'] = arr_qtls[:, i]
            nfeats += [f'{col}_qtl{qs_loc:.2f}']
    return featd, nfeats


def perform_quantile_bydscode(featd, feats=[], qs=[], folder=None, name=None, r=g_reg):
    """
    we use an expanding quantile computation
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert len(qs) > 0
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_qtl = QuantileStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_qtl", qs=qs)
        arr_qtls = cls_qtl.update(featd['dtsi'], featd['dscode'], featd[col])
        r.add(cls_qtl)
        for i in range(len(qs)):
            qs_loc = qs[i]
            featd[f'{col}_qtl{qs_loc:.2f}'] = arr_qtls[:, i]
            nfeats += [f'{col}_qtl{qs_loc:.2f}']
    return featd, nfeats


def perform_clip_quantile_global(featd, feats=[], folder=None, name=None, low_clip=0.05, high_clip=0.95, r=g_reg):
    """
    we use an expanding quantile computation
    This is still quite slow unfortunately !! 
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_qtl = QuantileStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_clip", qs=[low_clip, high_clip])
        arr_qtls = cls_qtl.update(featd['dtsi'], np.zeros(featd['dtsi'].shape[0], dtype=np.int64), featd[col])
        arr_qtl_low = arr_qtls[:, 0]
        arr_qtl_high = arr_qtls[:, 1]
        featd[f'{col}_qtllow'] = arr_qtl_low
        featd[f'{col}_qtlhigh'] = arr_qtl_high
        featd[f'{col}_clipqtl'] = np.where(featd[col] < arr_qtl_low,
                                           arr_qtl_low,
                                           np.where(
            featd[col] > arr_qtl_high,
            arr_qtl_high,
            featd[col]
        ))
        nfeats += [f'{col}_clipqtl']
    return featd, nfeats


def perform_cast_float64(featd, feats=[], folder=None, name=None, r=g_reg):
    """
    we use an expanding quantile computation
    This is still quite slow unfortunately !! 
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    for col in feats:
        featd[col] = featd[col].astype(np.float64, copy=False)
    return featd, feats


def perform_0tonan(featd, feats=[], folder=None, name=None, r=g_reg):
    """
    the alpha is variable , it depends on the ufeats
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        featd[f'{col}_0tonan'] = np.where(featd[col] == 0, np.nan, featd[col])
        nfeats += [f'{col}_0tonan']
    return featd, nfeats


def perform_add_prefix(featd, feats=[], prefix='', folder=None, name=None, r=g_reg):
    nfeats = []
    for col in feats:
        assert col in featd.keys()
        featd[f'{prefix}_{col}'] = featd.pop(col)
        nfeats += [f'{prefix}_{col}']
    return featd, nfeats


def perform_pfp(featd, feats=[], nbrevs=[1], ticks=[3.0],
                windows=[50, 100, 500, 1000],
                debug=False, folder=None, name=None, r=g_reg):
    """
    feats must be prices
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nbrevs = np.int64(nbrevs)
    nfeats = []
    for col in feats:
        for nbrev in nbrevs:
            for tick in ticks:
                ncol = f'{col}_pfp{tick}x{nbrev}'
                assert np.sum(np.isnan(featd[col])) == 0
                cls_pfp = PfPStepper \
                    .load(folder=f"{folder}", name=f"{name}_{ncol}", nbrev=nbrev, tick=tick)
                pfp_price, pfp_dir, pfp_el, pfp_perf, pfp_perf2, pfp_dur = \
                    cls_pfp.update(featd['dtsi'], featd['dscode'], featd[col])
                r.add(cls_pfp)
                featd[f'{ncol}_px'] = pfp_price
                featd[f'{ncol}_el'] = pfp_el
                featd[f'{ncol}_dir'] = pfp_dir
                featd[f'{ncol}_perf'] = pfp_perf
                featd[f'{ncol}_perf2'] = pfp_perf2
                featd[f'{ncol}_dur'] = pfp_dur
                # adding the diff of the dir
                cls_diff = DiffStepper \
                    .load(folder=f"{folder}", name=f"{name}_{ncol}_dir_diff", window=1)
                val_diff = cls_diff.update(featd['dtsi'], featd['dscode'], featd[f'{ncol}_dir'])
                r.add(cls_diff)
                featd[f'{ncol}_dir_chg'] = np.sign(np.abs(val_diff))
                nfeats += [f'{ncol}_dir', f'{ncol}_dir_chg',
                           f'{ncol}_perf2', f'{ncol}_dur']
                # and some ewm of the dir_chg
                for halflife in windows:
                    cls_ewm = EwmStepper \
                        .load(folder=f"{folder}", name=f"{name}_{ncol}_dir_diff_ewm{halflife}", window=halflife)
                    val_diff_ewm = cls_ewm.update(featd['dtsi'], featd['dscode'], featd[f'{ncol}_dir_chg'])
                    r.add(cls_ewm)
                    featd[f'{ncol}_dir_chg_ewm{halflife}'] = val_diff_ewm
                    nfeats += [f'{ncol}_dir_chg_ewm{halflife}']
    return featd, nfeats


def perform_cs_appops(featd, feats=[], windows=[1000], folder=None, name=None, r=g_reg):
    """same as a rolling rank but we do not pass the dscode. 
    Meaning that it happens to work cross - sectionally as well
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert 'univ' in featd.keys()
    assert len(windows) > 0
    assert np.all(np.diff(featd['dtsi']) >= 0)

    ndscode = featd['univ']

    nfeats = []
    for col in feats:
        for win in windows:
            cls_qtl = BottleneckRankStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_appops{win}", window=win)
            featd[f'{col}_appops{win}'] = cls_qtl.update(featd['dtsi'], ndscode, featd[col])
            r.add(cls_qtl)
            nfeats += [f'{col}_appops{win}']
    return featd, nfeats


def perform_cs_rank(featd, feats=[], folder=None, name=None, r=g_reg):
    """rank is between -1 and 1 here"""
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_qtl = csRankStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_csrank", percent=0)
        featd[f'{col}_csrank'] = cls_qtl.update(featd['dtsi'], featd['dscode'], featd[col])
        r.add(cls_qtl)
        nfeats += [f'{col}_csrank']
    return featd, nfeats


def perform_cs_rank_int(featd, feats=[], folder=None, name=None, r=g_reg):
    """returns an integer of the rank"""
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_qtl = csRankStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_csrank", percent=1)
        featd[f'{col}_csrank'] = cls_qtl.update(featd['dtsi'], featd['dscode'], featd[col])
        r.add(cls_qtl)
        nfeats += [f'{col}_csrank']
    return featd, nfeats


def perform_cs_rank_int_decreasing(featd, feats=[], folder=None, name=None, r=g_reg):
    """returns an integer of the rank but descending order"""
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_qtl = csRankStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_csrank", percent=2)
        featd[f'{col}_csrank'] = cls_qtl.update(featd['dtsi'], featd['dscode'], featd[col])
        r.add(cls_qtl)
        nfeats += [f'{col}_csrank']
    return featd, nfeats


def perform_model(featd, feats=[], wgt=None, ycol=None, folder=None, name=None,
                  lookback=300, minlookback=100,
                  fitfreq=10, gap=1, model_gen=None,
                  with_fit=True, r=g_reg):
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    check_cols(featd, feats)
    assert np.all(np.diff(featd['dtsi']) >= 0)
    xcols = get_hash('_'.join(feats))[:8]
    cls_model = ModelStepper \
        .load(folder=f"{folder}",
              name=f"{name}_model_{xcols}_{wgt}_{ycol}",
              lookback=lookback,
              minlookback=minlookback,
              fitfreq=fitfreq,
              gap=gap,
              model_gen=model_gen,
              with_fit=with_fit,
              featnames=feats)
    xseries = np.transpose(np.stack([v for k, v in featd.items() if k in feats]))
    wgtserie = featd[wgt]
    yserie = featd[ycol]
    res = cls_model.update(featd['dtsi'], xseries, yserie=yserie, wgtserie=wgtserie)
    r.add(cls_model)
    featd[f'model_{ycol}_{wgt}'] = res
    nfeats = [f'model_{ycol}_{wgt}']
    return featd, nfeats


def perform_model_batch(featd, feats=[], wgt=None, ycol=None, folder=None, name=None,
                        lookback=300, minlookback=100, ramlookback=10,
                        batch_size=300, lr=1e-3, epochs=10, weight_decay=1e-3,
                        fitfreq=10, gap=1, model_gen=None,
                        with_fit=True, r=g_reg):

    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    check_cols(featd, feats)
    assert np.all(np.diff(featd['dtsi']) >= 0)
    xcols = get_hash('_'.join(feats))[:8]
    cls_model = ModelBatchStepper \
        .load(folder=f"{folder}",
              name=f"{name}_model_batch_{xcols}_{wgt}_{ycol}",
              lookback=lookback,
              ramlookback=ramlookback,
              minlookback=minlookback,
              epochs=epochs,
              batch_size=batch_size,
              weight_decay=weight_decay,
              lr=lr,
              fitfreq=fitfreq,
              gap=gap,
              model_gen=model_gen,
              with_fit=with_fit,
              featnames=feats)
    xseries = np.transpose(np.stack([v for k, v in featd.items() if k in feats]))
    wgtserie = featd[wgt]
    yserie = featd[ycol]
    res = cls_model.update(featd['dtsi'], xseries, yserie=yserie, wgtserie=wgtserie)
    r.add(cls_model)
    featd[f'modelb_{ycol}_{wgt}'] = res
    nfeats = [f'modelb_{ycol}_{wgt}']
    return featd, nfeats


def perform_corr(featd, feats1=[], feats2=[], windows=[100], folder=None, name=None, r=g_reg):
    """
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    check_cols(featd, feats1)
    check_cols(featd, feats2)
    assert np.all(np.diff(featd['dtsi']) >= 0)
    assert np.array(windows).dtype == 'int64'
    nfeats = []
    for col1 in feats1:
        for col2 in feats2:
            for hl in windows:
                ncol = f'{col1}x{col2}_corr{hl}'
                cls_corr = RollingCorrStepper \
                    .load(folder=f"{folder}", name=f"{name}_corr_{ncol}", window=hl)
                corr_val = cls_corr.update(featd['dtsi'], featd['dscode'], featd[col1], featd[col2])
                r.add(cls_corr)
                featd[f'{ncol}'] = corr_val
                nfeats += [f'{ncol}']
    return featd, nfeats


def perform_cs_demean(featd, feats=[], by=None, wgt=None, folder=None, name=None, r=g_reg):
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_qtl = csMeanStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_csmean{by}{wgt}")
        csmean = cls_qtl.update(featd['dtsi'], featd['dscode'], featd[col],
                                by=None if by is None else featd[by],
                                wgt=None if wgt is None else featd[wgt])
        r.add(cls_qtl)
        featd[f'{col}_csmean'] = csmean
        featd[f'{col}_csdemean'] = featd[col]-csmean
        nfeats += [f'{col}_csdemean']
    return featd, nfeats


def perform_cs_scaling(featd, feats=[], by=None, wgt=None, folder=None, name=None, r=g_reg):
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_qtl = csStdStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_csstd{by}{wgt}")
        csstd = cls_qtl.update(featd['dtsi'], featd['dscode'], featd[col],
                               by=None if by is None else featd[by],
                               wgt=None if wgt is None else featd[wgt])
        r.add(cls_qtl)
        featd[f'{col}_csscaling'] = np.divide(
            featd[col],
            csstd,
            out=np.zeros_like(csstd),
            where=~np.isclose(csstd, np.zeros_like(csstd)))
        featd[f'{col}_csscaling'] = np.clip(featd[f'{col}_csscaling'], a_min=-3, a_max=3)
        nfeats += [f'{col}_csscaling']
    return featd, nfeats


def perform_cs_zscore(featd, feats=[], by=None, wgt=None, folder=None, name=None, r=g_reg):
    """ in a cross sectional way we do
    step1 = X - avg(X)
    step2 = step1 / std(step1)
    step3 = clip(step2,-3,3)
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_mean = csMeanStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_csmean{by}{wgt}")
        csmean = cls_mean.update(featd['dtsi'], featd['dscode'], featd[col],
                                 by=None if by is None else featd[by],
                                 wgt=None if wgt is None else featd[wgt])
        r.add(cls_mean)
        serie_demean = featd[col]-csmean

        cls_qtl = csStdStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_csstd{by}{wgt}")
        csstd = cls_qtl.update(featd['dtsi'], featd['dscode'], serie_demean,
                               by=None if by is None else featd[by],
                               wgt=None if wgt is None else featd[wgt])
        r.add(cls_qtl)
        featd[f'{col}_cszscore'] = np.divide(
            serie_demean,
            csstd,
            out=np.zeros_like(csstd),
            where=~np.isclose(csstd, np.zeros_like(csstd)))
        featd[f'{col}_cszscore'] = np.clip(featd[f'{col}_cszscore'], a_min=-3, a_max=3)
        nfeats += [f'{col}_cszscore']
    return featd, nfeats


def perform_reg(featd, feats1=[], feats2=[], windows=[100], lams=[0.0], folder=None, name=None, r=g_reg):
    """
    lam is the ridge regularisation
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    check_cols(featd, feats1)
    check_cols(featd, feats2)
    assert np.all(np.diff(featd['dtsi']) >= 0)
    assert np.array(windows).dtype == 'int64'
    nfeats = []
    for col1 in feats1:
        for col2 in feats2:
            for hl in windows:
                for lam in lams:
                    ncol = f'{col1}x{col2}_reg{hl}x{lam}'
                    cls_reg = RollingRidgeStepper \
                        .load(folder=f"{folder}", name=f"{name}_reg{ncol}", window=hl, lam=lam)
                    alpha, beta, resid = cls_reg.update(featd['dtsi'], featd['dscode'], featd[col1], featd[col2])
                    r.add(cls_reg)
                    featd[f'{ncol}_alpha'] = alpha
                    featd[f'{ncol}_beta'] = beta
                    featd[f'{ncol}_resid'] = resid
                    nfeats += [f'{ncol}_alpha', f'{ncol}_beta', f'{ncol}_resid']
    return featd, nfeats


def perform_pivot(featd, feats=[],  folder=None, name=None, r=g_reg):
    """
    returns date and dict { stock:values}
    Format is special
    """
    assert len(feats) == 1
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    check_cols(featd, feats)
    assert np.all(np.diff(featd['dtsi']) >= 0)
    col = feats[0]
    cls_piv = PivotStepper \
        .load(folder=f"{folder}", name=f"{name}_pivot_{col}")
    udts, res = cls_piv.update(featd['dtsi'], featd['dscode'], featd[col])
    r.add(cls_piv)
    return udts, res


def perform_unpivot(dts, pfeatd,  folder=None, name=None, r=g_reg):
    """
    returns date and dict { stock:values}
    Format is special
    """
    assert np.all(np.diff(dts) >= 0)
    cls_piv = UnPivotStepper \
        .load(folder=f"{folder}", name=f"{name}_unpivot")
    ndt, ndscode, nserie = cls_piv.update(dts, pfeatd)
    r.add(cls_piv)
    return ndt, ndscode, nserie


def perform_drawdown(featd, sigcol='', pnlcol='', folder=None, name=None, r=g_reg):
    """
    timeserie drawdown per dscode
    - we need sigcol to know the gross delta
    - we need windown to smooth the gross delta
    - drawdown is a percentage of the gross delta *100

    so condition on dd<5 for less than 5% drawdown
    """
    absserie = np.abs(featd[sigcol])
    cls_mean = ExpandingMeanStepper \
        .load(folder=f"{folder}", name=f"{name}_{sigcol}_expmean")
    gmvmean = cls_mean.update(featd['dtsi'], featd['dscode'], absserie)
    r.add(cls_mean)

    cls_cum = CumSumStepper \
        .load(folder=f"{folder}", name=f"{name}_{sigcol}_csum")
    cpnl = cls_cum.update(featd['dtsi'], featd['dscode'], featd[pnlcol])
    r.add(cls_cum)

    cls_emax = MaxStepper \
        .load(folder=f"{folder}", name=f"{name}_{sigcol}_csum_emax")
    cpnlmax = cls_emax.update(featd['dtsi'], featd['dscode'], cpnl)
    r.add(cls_emax)

    dd = cpnlmax-cpnl
    # dividing by the avg gmv to get a percentage number

    ddpct = np.divide(
        dd,
        gmvmean,
        out=np.zeros_like(gmvmean),
        where=~np.isclose(gmvmean,
                          np.zeros_like(gmvmean)))
    featd[f'{sigcol}_ddpct'] = ddpct * 100.0
    nfeats = [f'{sigcol}_ddpct']
    return featd, nfeats


def perform_bktest(featd,  with_plot=True, with_txt=True, commbps=1.0, folder=None, name=None, r=g_reg):
    """
    commbps is the commisson cost
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    cls_bk = BktestStepper \
        .load(folder=f"{folder}", name=f"{name}_bktest", commbps=commbps)
    cls_bk.update(featd)
    r.add(cls_bk)
    return cls_bk


def perform_avg_features_fillna0(featd, xcols=[], outname='', folder=None, name=None, r=g_reg):
    """we fillna values with 0.0
    avg(feat1,feat2,feat3,...featn)
    similar to df.mean(axis=1)
    """
    assert 'dtsi' in featd.keys()
    featd[outname] = np.zeros(featd['dtsi'].shape[0])
    for xcol in xcols:
        featd[outname] = featd[outname]+np.nan_to_num(featd[xcol])
    featd[outname] = featd[outname]/len(xcols)
    return featd, [outname]


def perform_bucketplot(featd, xcols=[], ycols=[],
                       n_buckets=8, freq=int(60*24*5),
                       folder=None, name=None, r=g_reg):
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    check_cols(featd, xcols+ycols)
    assert np.all(np.diff(featd['dtsi']) >= 0)
    for xcol in xcols:
        for ycol in ycols:
            cls_bucket = BucketXYStepper \
                .load(folder=f"{folder}",
                      name=f"{name}_bucketxy_{xcol}_{ycol}",
                      n_buckets=n_buckets,
                      freq=freq)
            r_mean, r_std = cls_bucket.update(featd['dtsi'],
                                              featd['dscode'],
                                              featd[xcol],
                                              featd[ycol])
            r.add(cls_bucket)
            featd[f'{xcol}_{ycol}_bucketxy_mean'] = r_mean
            featd[f'{xcol}_{ycol}_bucketxy_std'] = r_std
            nfeats = [
                f'{xcol}_{ycol}_bucketxy_mean',
                f'{xcol}_{ycol}_bucketxy_std']
    return featd, nfeats


def perform_clean_memory(featd, folder=None, name=None, r=g_reg):
    """
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    keep_cols = ['dtsi', 'dscode', 'wgt', 'univ', 'close']
    keep_cols += get_sig_cols(featd)
    keep_cols += get_sigf_cols(featd)

    for col in featd.keys():
        if not col in keep_cols:
            featd.pop(col)
    return featd, []


# ipython -i -m featurelib.lib_v1
if __name__ == '__main__':
    import numpy as np

    # Sample test data
    test_featd = {
        "dtsi": np.array([1, 2, 3, 4, 5], dtype=np.int64),  # Ensures sorted timestamps
        "dscode": np.array([1, 1, 1, 1, 1]),  # Asset code
        "price": np.array([100, 101, 103, 107, 110], dtype=np.float64),  # Example feature
        "volume": np.array([10, 12, 15, 20, 25], dtype=np.float64),  # Another feature
    }

    # Parameters for testing
    test_feats = ["price", "volume"]  # Features to compute diff
    test_windows = [1, 2]  # Windows for differencing
    test_folder = "test_folder"
    test_name = "test_diff"

    # Call the function
    output_featd, output_nfeats = perform_diff(
        featd=test_featd,
        feats=test_feats,
        windows=test_windows,
        folder=test_folder,
        name=test_name
    )

    # Print the results
    print("Updated Dictionary:")
    for key, value in output_featd.items():
        print(f"{key}: {value}")

    print("\nNew Features Created:", output_nfeats)
