import numpy as np

from stepper import *
from utils.common import clean_folder


g_steppers={} # list of sessions steppers 

def keep_import():
    clean_folder()


def check_cols(featd, wcols):
    for col in wcols:
        if not col in featd.keys():
            print(f'Missing col {col} in featd')
            assert False

def perform_save():
    """must be called before shutting down the ipython
    so that on restart we can continue from where we were
    """
    for k,step in g_steppers.items():
        step.save()

def add_to_stepper_register(stepper):
    # requires every stepper to have a __hash__ method
    if hash(stepper) in g_steppers:
        return 
    g_steppers[hash(stepper)]=stepper
    

def perform_ewm(featd, feats=[], windows=[1], folder=None, name=None):
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
            featd[f'{col}_ewm{hl}'] = ewm_val
            nfeats += [f'{col}_ewm{hl}']
            # keeping track for saving down before shut-down
            add_to_stepper_register(cls_ewm)
    return featd, nfeats


def perform_ewm_unit(featd, feats=[], ufeats=[], windows=[1], folder=None, name=None):
    """
    the alpha is variable , it depends on the ufeats
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    for col in ufeats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        for ucol in ufeats:
            for hl in windows:
                cls_ewm = EwmUnitStepper \
                    .load(folder=f"{folder}", name=f"{name}_{col}x{ucol}_ewm{hl}", window=hl)
                ewm_val = cls_ewm.update(featd['dtsi'], featd['dscode'], featd[col], featd[ucol])
                featd[f'{col}x{ucol}_ewm{hl}'] = ewm_val
                nfeats += [f'{col}x{ucol}_ewm{hl}']
    return featd, nfeats


def perform_conv_kernel_spike(featd, feats=[], ufeats=[],
                              windows=[1],
                              spike_intervals=[1],
                              half_lifes=[100],
                              spike_widths=[50],
                              folder=None, name=None):
    """
    the alpha is variable , it depends on the ufeats
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    for col in feats:
        assert col in featd.keys()
    for col in ufeats:
        assert col in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for col in feats:
        for ucol in ufeats:
            for window in windows:
                for spike_width in spike_widths:
                    for half_life in half_lifes:
                        for spike_interval in spike_intervals:
                            ncol = f'kernel_spike_{col}x{ucol}_{window}x{half_life}x{spike_interval}x{spike_width}'
                            cls_k = RollingKernelTwapStepper \
                                .load(folder=f"{folder}", name=f"{name}_{ncol}",
                                      window=window,
                                      spike_interval=spike_interval,
                                      spike_width=spike_width,
                                      half_life=half_life)
                            k_val = cls_k.update(featd['dtsi'], featd['dscode'], featd[col], featd[ucol])
                            featd[ncol] = k_val
                            nfeats += [ncol]
    return featd, nfeats


def perform_ewm_std(featd, feats=[], windows=[1], folder=None, name=None):
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
            featd[f'{col}_ewmstd{hl}'] = ewmstd_val
            nfeats += [f'{col}_ewmstd{hl}']
    return featd, nfeats


def perform_scaling_ewm(featd, feats=[], windows=[100], clip=3, folder=None, name=None):
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
            num = featd[col]
            denum = ewmstd_val
            featd[f'{col}_scaled_ewmstd{hl}'] = np.divide(
                num,
                denum,
                out=np.zeros_like(denum),
                where=~np.isclose(denum, np.zeros_like(denum)))
            nfeats += [f'{col}_scaled_ewmstd{hl}']
    return featd, nfeats


def perform_detrend_ewm(featd, feats=[], windows=[1], folder=None, name=None):
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
            featd[f'{col}_detrend_ewm{hl}'] = featd[col] - ewm_val
            nfeats += [f'{col}_detrend_ewm{hl}']
    return featd, nfeats


def perform_detrend_ewm_ratio(featd, feats=[], windows=[1], folder=None, name=None):
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
                .load(folder=f"{folder}", name=f"{name}_{col}_detrendratio_ewm{hl}", window=hl)
            ewm_val = cls_ewm.update(featd['dtsi'], featd['dscode'], featd[col])
            num = featd[col]
            denum = ewm_val
            featd[f'{col}_detrendratio_ewm{hl}'] = np.divide(
                num,
                denum,
                out=np.zeros_like(denum),
                where=~np.isclose(denum, np.zeros_like(denum))
            )
            nfeats += [f'{col}_detrendratio_ewm{hl}']
    return featd, nfeats


def perform_ewm_skew(featd, feats=[], windows=[1], folder=None, name=None):
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
            featd[f'{col}_ewmskew{hl}'] = skew_val
            nfeats += [f'{col}_ewmskew{hl}']
    return featd, nfeats


def perform_ewm_kurt(featd, feats=[], windows=[1], folder=None, name=None):
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
            featd[f'{col}_ewmkurt{hl}'] = kurt_val
            nfeats += [f'{col}_ewmkurt{hl}']
    return featd, nfeats


def perform_ffill(featd, feats=[], folder=None, name=None):
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
        featd[f'{col}_ffill'] = ffill_val
        nfeats += [f'{col}_ffill']
    return featd, nfeats


def perform_groupby_last(featd, feats=[], folder=None, name=None):
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
        if 'dtsi' not in nfeatd.keys():
            nfeatd['dtsi'] = rts
            nfeatd['dscode'] = rcode
        nfeatd[col] = rval
        nfeats += [col]
    return nfeatd, nfeats


def perform_cumsum(featd, feats=[], folder=None, name=None):
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
        featd[f'{col}_cumsum'] = cum_val
        nfeats += [f'{col}_cumsum']
    return featd, nfeats


def perform_diff(featd, feats=[], windows=[1], folder=None, name=None):
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
            featd[f'{col}_diff{hl}'] = np.nan_to_num(diff_val)
            nfeats += [f'{col}_diff{hl}']
    return featd, nfeats


def perform_lag(featd, feats=[], windows=[1], folder=None, name=None):
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
                featd[f'{col}_lag{hl}'] = lag_val
                nfeats += [f'{col}_lag{hl}']
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
            .load(folder=f"{folder}", name=f"{name}_{col}_masof", p=10)
        merge_val = cls_merge.update(featd_l['dtsi'], featd_l['dscode'],
                                     featd_r['dtsi'], featd_r['dscode'], featd_r[col])
        featd_l[f'{col}_masof'] = merge_val
        nfeats += [f'{col}_masof']
    return featd_l, nfeats


def perform_clip(featd, feats=[], clip=1.0, folder=None, name=None):
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
        featd[f'{col}_clip{clip}'] = np.clip(featd[col], -clip, clip)
        nfeats += [f'{col}_clip{clip}']
    return featd, nfeats


def perform_0tonan(featd, feats=[], folder=None, name=None):
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


def perform_add_prefix(featd, feats=[], prefix='', folder=None, name=None):
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


def perform_corr(featd, feats1=[], feats2=[], windows=[100], folder=None, name=None):
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
                featd[f'{ncol}'] = corr_val
                nfeats += [f'{ncol}']
    return featd, nfeats


def perform_reg(featd, feats1=[], feats2=[], windows=[100], lams=[0.0], folder=None, name=None):
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
                    featd[f'{ncol}_alpha'] = alpha
                    featd[f'{ncol}_beta'] = beta
                    featd[f'{ncol}_resid'] = resid
                    nfeats += [f'{ncol}_alpha', f'{ncol}_beta', f'{ncol}_resid']
    return featd, nfeats


def perform_orderbook_bbo(featd, windows=[100], lams=[0.0], folder=None, name=None):
    """
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert 'bidask' in featd.keys()
    assert 'price' in featd.keys()
    assert 'quantity' in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    cls_bbo = OrderBookStepper \
        .load(folder=f"{folder}", name=f"{name}_bbo")
    rbid, rask = cls_bbo.update(featd['dtsi'], featd['dscode'],
                                featd['bidask'], featd['price'],
                                featd['quantity'])
    featd[f'bbo_bid'] = rbid
    featd[f'bbo_ask'] = rask
    nfeats += ['bbo_bid', 'bbo_ask']
    return featd, nfeats


def perform_orderbook_imb_naive(featd, levels=[100], folder=None, name=None):
    """
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert 'bidask' in featd.keys()
    assert 'price' in featd.keys()
    assert 'quantity' in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for level in levels:
        cls_bbo = OrderBookStepper \
            .load(folder=f"{folder}", name=f"{name}_imb_{level}", op='imb', window=level)
        rbid, rask = cls_bbo.update(featd['dtsi'], featd['dscode'],
                                    featd['bidask'], featd['price'],
                                    featd['quantity'])
        num = rask - rbid
        denum = rask + rbid
        featd[f'ob_imb{level}'] = np.divide(
            num,
            denum,
            out=np.zeros_like(denum),
            where=~np.isclose(denum, np.zeros_like(denum))
        )

        nfeats += [f'ob_imb{level}']
    return featd, nfeats


def perform_orderbook_nonzero_levels(featd, levels=[100], folder=None, name=None):
    """
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert 'bidask' in featd.keys()
    assert 'price' in featd.keys()
    assert 'quantity' in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for level in levels:
        cls_bbo = OrderBookStepper \
            .load(folder=f"{folder}", name=f"{name}_nonzero_{level}", op='sum', window=level, use_sign=True)
        rbid, rask = cls_bbo.update(featd['dtsi'], featd['dscode'],
                                    featd['bidask'], featd['price'],
                                    featd['quantity'])
        featd[f'ob_nonzero{level}_bid'] = rbid
        featd[f'ob_nonzero{level}_ask'] = rask
        nfeats += [f'ob_nonzero{level}_bid', f'ob_nonzero{level}_ask']
    return featd, nfeats


def perform_orderbook_depth_func(featd, levels=[100], op='depth', folder=None, name=None):
    """
    how many orders there are
    """
    assert 'dtsi' in featd.keys()
    assert 'dscode' in featd.keys()
    assert 'bidask' in featd.keys()
    assert 'price' in featd.keys()
    assert 'quantity' in featd.keys()
    assert np.all(np.diff(featd['dtsi']) >= 0)
    nfeats = []
    for level in levels:
        cls_bbo = OrderBookDepthStepper \
            .load(folder=f"{folder}", name=f"{name}_{op}_{level}",
                  op=op,
                  window=level)
        rbid, rask = cls_bbo.update(featd['dtsi'], featd['dscode'],
                                    featd['bidask'], featd['price'],
                                    featd['quantity'])
        featd[f'ob_{op}_bid_l{level}'] = rbid
        featd[f'ob_{op}_ask_l{level}'] = rask
        num = rask - rbid
        denum = rask + rbid
        featd[f'ob_{op}_l{level}'] = np.divide(
            num,
            denum,
            out=np.zeros_like(denum),
            where=~np.isclose(denum, np.zeros_like(denum))
        )
        nfeats += [
            f'ob_{op}_bid_l{level}',
            f'ob_{op}_ask_l{level}',
            f'ob_{op}_l{level}']
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
    