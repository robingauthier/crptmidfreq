from crptmidfreq.stepper import *

from crptmidfreq.featurelib.lib_v1 import *
from crptmidfreq.utils.common import merge_dicts
from crptmidfreq.utils_mr.pairs_sel import pick_k_pairs


def pairs_feats(featd, incol='mual', fretcol='tret_xmkt', 
              folder=None, name=None, r=None, cfg={}):
    assert fretcol in featd.keys()
    dcfg = dict(
        windows_sharpe=[1000, 5000],
        window_appops=3000
    )
    cfg = merge_dicts(cfg, dcfg, name='pnl_feats')
    
    pdts, pfeatd = perform_pivot(featd=featd,
                                 feats=[incol],
                                 **defargs)

    pdts, puniv = perform_pivot(featd=featd,
                                feats=['univ'],
                                **defargs)
    clusterd = puniv
    cls = CorrelDistanceStepper(
        lookback=300, 
        fitfreq=10,
        folder=folder,
        name=None,
        r=r
    )
    cls.update(pdts, pfeatd, clusterd)
    
    pick_k_pairs()
    
    import pdb;pdb.set_trace()
    


if __name__=='__main__':
    from crptmidfreq.strats.prepare_klines import prepare_klines
    from crptmidfreq.utils.univ import hardcoded_universe_1
    from crptmidfreq.stepper.zregistry import StepperRegistry
    g_r=StepperRegistry()
    cfg={}
    defargs = {'folder': 'test_pairs', 'name': None, 'r': g_r, 'cfg': cfg}
    # read the data from the DuckDB
    featd = prepare_klines(start_date='2025-01-01',
                           end_date='2025-03-01',
                           tokens=hardcoded_universe_1,
                           **defargs)
    featd=pairs_feats(featd,**defargs)