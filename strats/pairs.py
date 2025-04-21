from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.stepper import *
from crptmidfreq.utils.common import merge_dicts
from crptmidfreq.utils.lazy_dict import LazyDict


def pairs_feats(featd, incol='tret_xmkt',
                folder=None, name=None, r=None, cfg={}):

    dcfg = dict(
        pairs_distance_lookback=300,
        pairs_distance_fitfreq=100,
    )
    cfg = merge_dicts(cfg, dcfg, name='pairs_feats')
    defargs = {'folder': folder, 'name': name, 'r': r}
    pdts, pfeatd = perform_pivot(featd=featd,
                                 feats=[incol],
                                 **defargs)

    pdts, puniv = perform_pivot(featd=featd,
                                feats=['univ'],
                                **defargs)
    clusterd = puniv
    cls_dst = CorrelDistanceStepper(
        lookback=cfg.get('pairs_distance_lookback'),
        fitfreq=cfg.get('pairs_distance_fitfreq'),
        folder=folder,
        name=None,
    )

    dstfeatd = cls_dst.update(pdts, pfeatd, clusterd)
    r.add(cls_dst)

    cls_k = PairsSelKStepper(
        folder=folder,
        name=None,
        k=3,
    )
    dstfeatd2 = cls_k.update(dstfeatd['dtsi'], dstfeatd['dscode1'], dstfeatd['dscode2'], dstfeatd['dist'])
    r.add(cls_k)

    cls_pairs = PairsExplodeStepper(
        folder=folder,
        name=None,
    )
    pairsfeatd = cls_pairs.update(featd['dtsi'], featd['dscode'], featd[incol],
                                  dstfeatd2['dtsi'], dstfeatd2['dscode1'], dstfeatd2['dscode2'])

    r.add(cls_pairs)

    # Putting all of this in a LazyDict
    rd = LazyDict()

    import pdb
    pdb.set_trace()


# ipython -i -m crptmidfreq.strats.pairs
if __name__ == '__main__':
    from crptmidfreq.stepper.zregistry import StepperRegistry
    from crptmidfreq.strats.prepare_klines import prepare_klines
    from crptmidfreq.utils.univ import hardcoded_universe_1
    g_r = StepperRegistry()
    cfg = {}
    defargs = {'folder': 'test_pairs', 'name': None, 'r': g_r, 'cfg': cfg}
    # read the data from the DuckDB
    featd = prepare_klines(start_date='2025-01-01',
                           end_date='2025-03-01',
                           tokens=hardcoded_universe_1,
                           **defargs)
    featd['univ'] = np.ones_like(featd['dtsi'], dtype=np.int64)
    featd = pairs_feats(featd, incol='tret', **defargs)
