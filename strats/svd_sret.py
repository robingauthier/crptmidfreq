from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.stepper.incr_svd import SVDStepper
from crptmidfreq.utils.common import merge_dicts, rename_key


def svd_sret(featd,
             incol='tret_xmkt',
             oucol='sret_svd',
             folder=None, name=None, r=None, cfg={}):
    """
    input on : incol
    and outputs : oucol
    """
    assert incol in featd.keys()
    assert 'univ' in featd.keys()
    dcfg = dict(
        svd_k=20,
        svd_lookback=10000,
        svd_fitfreq=100,

        sret_clip=0.01,  # 3%
    )
    cfg = merge_dicts(cfg, dcfg, name='kmeans_sret')
    # arguments always used
    defargs = {'folder': folder, 'r': r, 'name': name}

    # pivotting for correlation matrix / clustering
    pdts, pfeatd = perform_pivot(featd=featd,
                                 feats=[incol],
                                 **defargs)

    pdts, puniv = perform_pivot(featd=featd,
                                feats=['univ'],
                                **defargs)

    # Fitting the model now / directly calling stepper
    cls_model = SVDStepper \
        .load(folder=folder,
              name=f'{name}_model_svd',
              lookback=cfg.get('svd_lookback'),
              fitfreq=cfg.get('svd_fitfreq'),
              n_comp=cfg.get('svd_k'),
              )
    resid_d, diag_arr = cls_model.update(pdts, pfeatd, puniv)
    r.add(cls_model)

    # res is a matrix ndst x ndscode
    ndt, ndscode, nserie = perform_unpivot(pdts, resid_d, **defargs)
    featd2 = {'dtsi': ndt, 'dscode': ndscode, 'sret_svd_1': nserie}

    # below is same as left merge
    featd, nfeats = perform_merge_asof(featd,
                                       featd2,
                                       feats=['sret_svd_1'],
                                       **defargs)

    featd = rename_key(featd, nfeats[0], oucol)

    # Adding the diag_arr for variance explained
    featd['all'] = np.ones_like(featd['dtsi'], dtype=np.int64)
    diag_d = {'dtsi': pdts,
              'all': np.ones_like(pdts, dtype=np.int64)
              }
    nfeats = []
    for k in range(cfg.get('svd_k')):
        diag_d[f'd_{k}'] = diag_arr[:, k]
        nfeats += [f'd_{k}']
    featd, nfeats = perform_merge_asof(featd,
                                       diag_d,
                                       key='all',
                                       feats=nfeats,
                                       **defargs)
    featd, _ = perform_to_sigf(featd, nfeats, **defargs)

    return featd
