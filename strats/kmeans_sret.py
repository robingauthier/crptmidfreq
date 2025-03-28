from sklearn.cluster import KMeans
from crptmidfreq.utils.common import rename_key
from numba.typed import Dict
from numba.core import types
from crptmidfreq.featurelib.lib_v1 import *

from crptmidfreq.utils.common import merge_dicts


def kmeans_sret(featd, incol='tret_xmkt', oucol='sret_kmeans',
                folder=None, name=None, r=None, cfg={}):
    """
    input on : incol
    and outputs : oucol
    """
    assert incol in featd.keys()
    dcfg = dict(
        kmeans_k=20,
        kmeans_lookback=10000,
        kmeans_fitfreq=100
    )
    cfg = merge_dicts(cfg, dcfg, name='kmeans_sret')

    # pivotting for correlation matrix / clustering
    pdts, pfeatd = perform_pivot(featd=featd,
                                 feats=[incol],
                                 folder=folder,
                                 name=name,
                                 r=r)
    pX = np.array([v for k, v in pfeatd.items()])
    pX = np.nan_to_num(pX)
    pX = np.transpose(pX)  # ndts x ndscode
    pdscodes = list(pfeatd.keys())

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
              with_fit=True)
    # kmeanres is 2dim array ndts x ndscode
    kmeansres = cls_model.update(pdts, pfeatd, yserie=None, wgtserie=None)
    r.add(cls_model)

    kmeansd = Dict.empty(
        key_type=types.int64,    # Define the type of keys
        value_type=types.Array(types.float64, 1, "C")  # Define the type of values
    )
    for i in range(len(pdscodes)):
        kmeansd[pdscodes[i]] = kmeansres[:, i].copy(order='C')

    # res is a matrix ndst x ndscode
    ndt, ndscode, nserie = perform_unpivot(pdts, kmeansd,
                                           folder=folder,
                                           name=name,
                                           r=r)
    featd2 = {'dtsi': ndt, 'dscode': ndscode, 'kmeans_cat': nserie}

    featd, nfeats = perform_merge_asof(featd,
                                       featd2,
                                       feats=['kmeans_cat'],
                                       folder=folder,
                                       name=name,
                                       r=r)
    featd = rename_key(featd, nfeats[0], 'kmeans_cat')

    # Now removing the cluster mean
    featd, nfeats = perform_cs_demean(featd=featd,
                                      feats=[incol],
                                      by='kmeans_cat',
                                      wgt='wgt',
                                      folder=folder,
                                      name=name,
                                      r=r)
    featd = rename_key(featd, nfeats[0], oucol)

    return featd
