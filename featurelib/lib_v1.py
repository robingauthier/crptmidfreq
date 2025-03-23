import os
import numpy as np
import pandas as pd
from crptmidfreq.stepper import *
from crptmidfreq.utils.common import clean_folder
from crptmidfreq.stepper.zregistry import StepperRegistry
from crptmidfreq.utils.common import get_analysis_folder
from crptmidfreq.utils.common import rename_key
from pprint import pprint

g_reg = StepperRegistry()

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
        

def perform_ewm(featd, feats=[], windows=[1], folder=None, name=None,r=g_reg):
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

def perform_divide(featd,numcols=[],denumcols=[],folder=None,name=None,r=g_reg):
    for col in numcols:
        assert col in featd.keys()
    for col in denumcols:
        assert col in featd.keys()
    nfeats=[]
    for numcol in numcols:
        for denumcol in denumcols:
            featd[f'{numcol}div{denumcol}'] = np.divide(
                    featd[numcol], 
                    featd[denumcol], 
                    out=np.zeros_like(featd[denumcol]),
                    where=~np.isclose(featd[denumcol], 
                                      np.zeros_like(featd[denumcol])))
            nfeats+=[f'{numcol}div{denumcol}']
    return featd, nfeats


def perform_to_sig(featd,feats=[],folder=None,name=None,r=g_reg):
    nfeats=[]
    for col in feats:
        featd=rename_key(featd,col,f'sig_{col}')
        nfeats+=['sig_'+col]
    return featd, nfeats


def perform_divide_m1(featd,numcols=[],denumcols=[],folder=None,name=None,r=g_reg):
    """removes one once we divided"""
    for col in numcols:
        assert col in featd.keys()
    for col in denumcols:
        assert col in featd.keys()
    nfeats=[]
    for numcol in numcols:
        for denumcol in denumcols:
            featd[f'{numcol}divm1{denumcol}'] = np.divide(
                    featd[numcol], 
                    featd[denumcol], 
                    out=np.ones_like(featd[denumcol]),
                    where=~np.isclose(featd[denumcol], 
                                      np.zeros_like(featd[denumcol])))
            featd[f'{numcol}divm1{denumcol}']=featd[f'{numcol}divm1{denumcol}']-1.0
            nfeats+=[f'{numcol}div{denumcol}']
    return featd, nfeats

def perform_ewm_std(featd, feats=[], windows=[1], folder=None, name=None,r=g_reg):
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


def perform_scaling_ewm(featd, feats=[], windows=[100], clip=3, folder=None, name=None,r=g_reg):
    """scaling by ewmstd and clipping"""
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
            featd[f'{col}_scaled_ewmstd{hl}']=np.clip(featd[f'{col}_scaled_ewmstd{hl}'],a_min=-clip,a_max=clip)
            nfeats += [f'{col}_scaled_ewmstd{hl}']
    return featd, nfeats


def perform_detrend_ewm(featd, feats=[], windows=[1], folder=None, name=None,r=g_reg):
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
            cls_ewm = EwmStepper \
                .load(folder=f"{folder}", name=f"{name}_{col}_detrend_ewm{hl}", window=hl)
            ewm_val = cls_ewm.update(featd['dtsi'], featd['dscode'], featd[col])
            r.add(cls_ewm)
            featd[f'{col}_detrend_ewm{hl}'] = featd[col] - ewm_val
            nfeats += [f'{col}_detrend_ewm{hl}']
    return featd, nfeats


def perform_detrend_ewm_ratio(featd, feats=[], windows=[1], folder=None, name=None,r=g_reg):
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
            featd[f'{col}_detrendratio_ewm{hl}']=featd[f'{col}_detrendratio_ewm{hl}']-1.0
            nfeats += [f'{col}_detrendratio_ewm{hl}']
    return featd, nfeats


def perform_ewm_skew(featd, feats=[], windows=[1], folder=None, name=None,r=g_reg):
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


def perform_ewm_kurt(featd, feats=[], windows=[1], folder=None, name=None,r=g_reg):
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


def perform_ffill(featd, feats=[], folder=None, name=None,r=g_reg):
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


def perform_groupby_last(featd, feats=[], folder=None, name=None,r=g_reg):
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


def perform_cumsum(featd, feats=[], folder=None, name=None,r=g_reg):
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


def perform_cnt_exists(featd, feats=[], folder=None, name=None,r=g_reg):
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


def perform_diff(featd, feats=[], windows=[1], folder=None, name=None,r=g_reg):
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


def perform_lag(featd, feats=[], windows=[1], folder=None, name=None,r=g_reg):
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

def perform_lag_forward(featd, feats=[], windows=[1], folder=None, name=None,r=g_reg):
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
            assert hl<0
            df = pd.DataFrame({
                'i':np.arange(len(featd['dtsi'])),
                'dts':featd['dtsi'],
                'dscode':featd['dscode'],
                'feat':featd[col],
                })
            df['feat_forward']=df.groupby('dscode')['feat'].transform(lambda x:x.shift(hl))
            assert df['i'].is_monotonic_increasing
            featd[f'forward_{col}_lag{hl}'] = df['feat_forward'].values
            nfeats += [f'forward_{col}_lag{hl}']
    return featd, nfeats


def perform_max(featd, feats=[], windows=[1], folder=None, name=None):
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


def perform_min(featd, feats=[], windows=[1], folder=None, name=None):
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


def perform_sma(featd, feats=[], windows=[1], folder=None, name=None):
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


def perform_merge_asof(featd_l, featd_r, feats=[], folder=None, name=None):
    """
    """
    assert 'dtsi' in featd_l.keys()
    assert 'dscode' in featd_l.keys()
    assert np.all(np.diff(featd_l['dtsi']) >= 0)

    assert 'dtsi' in featd_r.keys()
    assert 'dscode' in featd_r.keys()
    assert np.all(np.diff(featd_r['dtsi']) >= 0)

    if len(feats) == 0:
        feats = list(featd_r.keys())
        feats = [x for x in feats if x != 'dtsi']
        feats = [x for x in feats if x != 'dscode']

    nfeats = []
    for col in feats:
        cls_merge = MergeAsofStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_masof")
        merge_val = cls_merge.update(featd_l['dtsi'], featd_l['dscode'],
                                     featd_r['dtsi'], featd_r['dscode'], featd_r[col])
        featd_l[f'{col}_masof'] = merge_val
        nfeats += [f'{col}_masof']
    return featd_l, nfeats


def perform_clip(featd, feats=[], folder=None, name=None,low_clip=np.nan,high_clip=np.nan):
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
            .load(folder=f"{folder}", name=f"{name}_{col}_clip", low_clip=low_clip,high_clip=high_clip)
        featd[f'{col}_clip'] = cls_clip.update(featd['dtsi'], featd['dscode'],featd[col])
        nfeats += [f'{col}_clip']
    return featd, nfeats

def perform_quantile_global(featd, feats=[], qs=[],folder=None, name=None,r=g_reg):
    """
    we use an expanding quantile computation
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert len(qs)>0
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_qtl = QuantileStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_qtl", qs=qs)
        arr_qtls = cls_qtl.update(featd['dtsi'], np.zeros(featd['dtsi'].shape[0],dtype=np.int64),featd[col])
        r.add(cls_qtl)
        for i in range(len(qs)):
            qs_loc = qs[i]
            featd[f'{col}_qtl{qs_loc:.2f}'] = arr_qtls[:,i]
            nfeats += [f'{col}_qtl{qs_loc:.2f}']
    return featd, nfeats

def perform_quantile_bydscode(featd, feats=[], qs=[],folder=None, name=None,r=g_reg):
    """
    we use an expanding quantile computation
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert len(qs)>0
    for col in feats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_qtl = QuantileStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_qtl", qs=qs)
        arr_qtls = cls_qtl.update(featd['dtsi'], featd['dscode'],featd[col])
        r.add(cls_qtl)
        for i in range(len(qs)):
            qs_loc = qs[i]
            featd[f'{col}_qtl{qs_loc:.2f}'] = arr_qtls[:,i]
            nfeats += [f'{col}_qtl{qs_loc:.2f}']
    return featd, nfeats

def perform_clip_quantile_global(featd, feats=[], folder=None, name=None,low_clip=0.05,high_clip=0.95,r=g_reg):
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
            .load(folder=f"{folder}", name=f"{name}_{col}_clip", qs=[low_clip,high_clip])
        arr_qtls = cls_qtl.update(featd['dtsi'], np.zeros(featd['dtsi'].shape[0],dtype=np.int64),featd[col])
        arr_qtl_low = arr_qtls[:,0]
        arr_qtl_high = arr_qtls[:,1]
        featd[f'{col}_qtllow']=arr_qtl_low
        featd[f'{col}_qtlhigh']=arr_qtl_high
        featd[f'{col}_clipqtl'] = np.where(featd[col]<arr_qtl_low,
                                            arr_qtl_low,
                                            np.where(
                                                featd[col]>arr_qtl_high,
                                            arr_qtl_high,
                                            featd[col]
                                            ))
        nfeats += [f'{col}_clipqtl']
    return featd, nfeats

def perform_cast_float64(featd, feats=[], folder=None, name=None,r=g_reg):
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

def perform_0tonan(featd, feats=[], folder=None, name=None,r=g_reg):
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


def perform_add_prefix(featd, feats=[], prefix='', folder=None, name=None,r=g_reg):
    nfeats = []
    for col in feats:
        assert col in featd.keys()
        featd[f'{prefix}_{col}'] = featd.pop(col)
        nfeats += [f'{prefix}_{col}']
    return featd, nfeats


def perform_pfp(featd, feats=[], nbrevs=[1], ticks=[3.0], debug=False, folder=None, name=None):
    """
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
                featd[f'{ncol}_dir'] = pfp_dir
                featd[f'{ncol}_perf'] = pfp_perf
                featd[f'{ncol}_perf2'] = pfp_perf2
                featd[f'{ncol}_dur'] = pfp_dur
                ## adding the diff of the dir
                cls_diff = DiffStepper \
                    .load(folder=f"{folder}", name=f"{name}_{ncol}_dir_diff", window=1)
                val_diff = cls_diff.update(featd['dtsi'], featd['dscode'], featd[f'{ncol}_dir'])
                featd[f'{ncol}_dir_chg'] = np.sign(np.abs(val_diff))
                nfeats += [f'{ncol}_dir', f'{ncol}_dir_chg',
                           f'{ncol}_perf', f'{ncol}_dur']
                ## and some ewm of the dir_chg
                for halflife in [50, 100, 500, 1000]:
                    cls_ewm = EwmStepper \
                        .load(folder=f"{folder}", name=f"{name}_{ncol}_dir_diff_ewm{halflife}", window=halflife)
                    val_diff_ewm = cls_ewm.update(featd['dtsi'], featd['dscode'], featd[f'{ncol}_dir_chg'])
                    featd[f'{ncol}_dir_chg_ewm{halflife}'] = val_diff_ewm
                    nfeats += [f'{ncol}_dir_chg_ewm{halflife}']
                if debug:
                    print('pfp :: Warning debug mode')
                    featd[f'{ncol}_px'] = pfp_price
                    featd[f'{ncol}_el'] = pfp_el
                    nfeats += [f'{ncol}_px', f'{ncol}_el']
    return featd, nfeats

def perform_cs_rank(featd,feats=[],folder=None, name=None,r=g_reg):
    """rank is between -1 and 1 here"""
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_qtl = csRankStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_csrank",percent=0)
        featd[f'{col}_csrank'] = cls_qtl.update(featd['dtsi'], featd['dscode'],featd[col])
        r.add(cls_qtl)
        nfeats += [f'{col}_csrank']
    return featd, nfeats

def perform_cs_rank_int(featd,feats=[],folder=None, name=None,r=g_reg):
    """returns an integer of the rank"""
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_qtl = csRankStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_csrank",percent=1)
        featd[f'{col}_csrank'] = cls_qtl.update(featd['dtsi'], featd['dscode'],featd[col])
        r.add(cls_qtl)
        nfeats += [f'{col}_csrank']
    return featd, nfeats

def perform_cs_rank_int_decreasing(featd,feats=[],folder=None, name=None,r=g_reg):
    """returns an integer of the rank but descending order"""
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_qtl = csRankStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_csrank",percent=2)
        featd[f'{col}_csrank'] = cls_qtl.update(featd['dtsi'], featd['dscode'],featd[col])
        r.add(cls_qtl)
        nfeats += [f'{col}_csrank']
    return featd, nfeats
    
def perform_model(featd, feats=[], wgt=None,ycol=None,folder=None, name=None,
                  lookback=300,minlookback=100,
                  fitfreq=10,gap=1,model_gen=None,
                  with_fit=True,r=g_reg):
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    check_cols(featd, feats)
    assert np.all(np.diff(featd['dtsi']) >= 0)
    xcols='_'.join(feats)
    cls_model = ModelStepper \
        .load(folder=f"{folder}", name=f"{name}_{xcols}_{wgt}_{ycol}", 
                lookback=lookback,minlookback=minlookback,
                 fitfreq=fitfreq,gap=gap,
                 model_gen=model_gen,with_fit=with_fit)
    xseries = np.concatenate([v for k,v in featd.items() if k in feats])
    wgtserie = featd[wgt]
    yserie = featd[ycol]
    res=cls_model.update(featd['dtsi'], xseries, yserie=yserie,wgtserie=wgtserie)
    r.add(cls_model)
    featd[f'model_{ycol}_{wgt}']=res
    nfeats=[f'model_{ycol}_{wgt}']
    return featd, nfeats

def perform_corr(featd, feats1=[], feats2=[], windows=[100], folder=None, name=None,r=g_reg):
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
                    .load(folder=f"{folder}", name=f"{name}_{ncol}", window=hl)
                corr_val = cls_corr.update(featd['dtsi'], featd['dscode'], featd[col1], featd[col2])
                r.add(cls_corr)
                featd[f'{ncol}'] = corr_val
                nfeats += [f'{ncol}']
    return featd, nfeats

def perform_cs_demean(featd,feats=[],by=None,wgt=None,folder=None, name=None,r=g_reg):
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_qtl = csMeanStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_csmean{by}{wgt}")
        csmean = cls_qtl.update(featd['dtsi'], featd['dscode'],featd[col],
                                by=None if by is None else featd[by],
                                wgt=None if wgt is None else featd[wgt])
        r.add(cls_qtl)
        featd[f'{col}_csdemean'] = featd[col]-csmean
        nfeats += [f'{col}_csdemean']
    return featd, nfeats

def perform_cs_scaling(featd,feats=[],by=None,wgt=None,folder=None, name=None,r=g_reg):
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        cls_qtl = csStdStepper \
            .load(folder=f"{folder}", name=f"{name}_{col}_csstd{by}{wgt}")
        csstd = cls_qtl.update(featd['dtsi'], featd['dscode'],featd[col],
                                by=None if by is None else featd[by],
                                wgt=None if wgt is None else featd[wgt])
        r.add(cls_qtl)
        featd[f'{col}_csscaling'] = np.divide(
                    featd[col],
                    csstd,
                    out=np.zeros_like(csstd),
                    where=~np.isclose(csstd, np.zeros_like(csstd)))
        nfeats += [f'{col}_csscaling']
    return featd, nfeats


def perform_reg(featd, feats1=[], feats2=[], windows=[100], lams=[0.0], folder=None, name=None,r=g_reg):
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
                        .load(folder=f"{folder}", name=f"{name}_{ncol}", window=hl, lam=lam)
                    alpha, beta, resid = cls_reg.update(featd['dtsi'], featd['dscode'], featd[col1], featd[col2])
                    r.add(cls_reg)
                    featd[f'{ncol}_alpha'] = alpha
                    featd[f'{ncol}_beta'] = beta
                    featd[f'{ncol}_resid'] = resid
                    nfeats += [f'{ncol}_alpha', f'{ncol}_beta', f'{ncol}_resid']
    return featd, nfeats


def perform_pivot(featd, feats=[],  folder=None, name=None,r=g_reg):
    """
    returns date and dict { stock:values}
    Format is special
    """
    assert len(feats)==1
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    check_cols(featd, feats)
    assert np.all(np.diff(featd['dtsi']) >= 0)
    col = feats[0]
    cls_piv = PivotStepper \
        .load(folder=f"{folder}", name=f"{name}_pivot_{col}")
    udts,res = cls_piv.update(featd['dtsi'], featd['dscode'], featd[col])
    r.add(cls_piv)
    return udts,res


def perform_unpivot(dts, pfeatd,  folder=None, name=None,r=g_reg):
    """
    returns date and dict { stock:values}
    Format is special
    """
    assert np.all(np.diff(dts) >= 0)
    cls_piv = UnPivotStepper \
        .load(folder=f"{folder}", name=f"{name}_unpivot")
    ndt,ndscode,nserie = cls_piv.update(dts, pfeatd)
    r.add(cls_piv)
    return ndt,ndscode,nserie


def perform_bktest(featd,  with_plot=True,with_txt=True,folder=None, name=None,r=g_reg):
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    from crptmidfreq.utils.bktester import bktest_stats
    if with_plot:
        save_graph_path=os.path.join(get_analysis_folder(),'bktest_NAME.png')
    else:
        save_graph_path = None
    sig_cols=[x for x in featd.keys() if x.startswith('sig_')]
    lr=[]
    for sig_col in sig_cols:
        stats=bktest_stats(
            featd['dtsi'],
            featd['dscode'],
            featd[sig_col],
            featd['forward_fh1'],
            featd['wgt'], 
            str(sig_col), # name
            save_graph_path=save_graph_path,
        )
        stats['col']=sig_col
        if with_txt:
            print('-'*20)
            print('-'*20)
            pprint(stats)
        lr+=[stats]
    if with_txt:
        rptdf=pd.DataFrame(lr)
        print('Gross P&L Stats:')
        rptdf1 = rptdf[['name','col','sr','rpt','mdd','rog','avg_gmv','ann_pnl']].round(2)
        print(rptdf1)
        print('Net Version:')
        rptdf2=rptdf[['name','col','sr_net','rpt_net','mdd_net','rog_net']]\
            .rename(columns=lambda x:x.replace('_net','')).round(2)
        print(rptdf2)
    return stats

def perform_avg_features_fillna0(featd,xcols=[],outname='',folder=None,name=None,r=g_reg):
    """we fillna values with 0.0"""
    assert 'dtsi' in featd.keys()
    featd[outname]=np.zeros(featd['dtsi'].shape[0])
    for xcol in xcols:
        featd[outname]=featd[outname]+np.nan_to_num(featd[xcol])
    featd[outname]=featd[outname]/len(xcols)
    return featd,[outname]

def perform_bucketplot(featd,xcols=[],ycols=[],folder=None,name=None,r=g_reg):
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    check_cols(featd, xcols+ycols)
    assert np.all(np.diff(featd['dtsi']) >= 0)
    for xcol in xcols:
        for ycol in ycols:
            cls_bucket = BucketXYStepper \
                .load(folder=f"{folder}", name=f"{name}_bucketxy_{xcol}_{ycol}")
            r_mean,r_std = cls_bucket.update(featd['dtsi'], 
                                             featd['dscode'], 
                                             featd[xcol],
                                             featd[ycol])
            # TODO: to finish
    nfeats=[]
    return featd, nfeats


# ipython -i -m featurelib.lib_v1
if __name__=='__main__':
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
    