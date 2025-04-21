from sklearn.cluster import KMeans

from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.utils.common import merge_dicts, rename_key


def kmeans_sret(featd, incol='tret_xmkt', oucol='sret_kmeans',
                folder=None, name=None, r=None, cfg={}):
    """
    input on : incol
    and outputs : oucol
    """
    assert incol in featd.keys()
    assert 'univ' in featd.keys()
    dcfg = dict(
        kmeans_k=20,
        kmeans_lookback=10000,
        kmeans_fitfreq=100,

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
    def model_gen_kmeans():
        return KMeans(n_clusters=cfg.get('kmeans_k'), random_state=42, n_init='auto')
    cls_model = PivotModelStepper \
        .load(folder=folder,
              name=f'{name}_model_kmeans',
              lookback=cfg.get('kmeans_lookback'),
              minlookback=500,
              fitfreq=cfg.get('kmeans_fitfreq'),
              gap=1,
              model_gen=model_gen_kmeans,
              is_kmeans=True,
              with_fit=True)
    # kmeanres is 2dim array ndts x ndscode
    kmeansd, kdts = cls_model.update(pdts, pfeatd, puniv)
    r.add(cls_model)

    # res is a matrix ndst x ndscode
    ndt, ndscode, nserie = perform_unpivot(kdts, kmeansd, **defargs)
    featd2 = {'dtsi': ndt, 'dscode': ndscode, 'kmeans_cat': nserie}

    featd, nfeats = perform_merge_asof(featd,
                                       featd2,
                                       feats=['kmeans_cat'],
                                       **defargs)
    featd = rename_key(featd, nfeats[0], 'kmeans_cat')

    # Now removing the cluster mean
    featd, nfeats = perform_cs_demean(featd=featd,
                                      feats=[incol],
                                      by='kmeans_cat',
                                      wgt='wgt',
                                      **defargs)

    # Clipping some large outliers
    th = cfg.get('sret_clip')
    featd, nfeats = perform_clip_quantile_global(featd=featd,
                                                 feats=nfeats,
                                                 low_clip=th,
                                                 high_clip=1-th,
                                                 folder=folder,
                                                 name=name,
                                                 r=r)

    featd = rename_key(featd, nfeats[0], oucol)
    return featd
