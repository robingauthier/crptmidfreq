import pandas as pd
import numpy as np
from crptmidfreq.stepper.base_stepper import BaseStepper
from crptmidfreq.utils.common import get_logger
from crptmidfreq.stepper.incr_model_timeclf import TimeClfStepper
from crptmidfreq.stepper.incr_model_timeclf import get_dts_max_before


logger = get_logger()


def featimp_lgbm(model_loc, feat_names):
    try:
        fimp = pd.Series(model_loc.feature_importances_, index=feat_names)
        print(fimp.sort_values(ascending=False))
    except Exception as e:
        pass


class ModelStepper(BaseStepper):
    """Relies on the pivot first and then runs an SVD"""

    def __init__(self, folder='', name='', lookback=300, minlookback=100,
                 fitfreq=10, gap=1, model_gen=None, with_fit=True, featnames=[]):
        """
        """
        super().__init__(folder, name)
        self.timeclf = TimeClfStepper(folder=folder, name=name,
                                      lookback=lookback, minlookback=minlookback,
                                      fitfreq=fitfreq, gap=gap)

        # Initialize empty state for the memory
        self.last_xmem = np.ndarray(shape=(0, 0), dtype=np.float64)
        self.last_ymem = np.array([], dtype=np.float64)
        self.last_wmem = np.array([], dtype=np.float64)
        self.last_dts = np.array([], dtype=np.int64)

        # History of models
        self.model_gen = model_gen
        self.hmodels = []

        self.with_fit = with_fit
        self.featnames = featnames

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, lookback=300, minlookback=100,
             fitfreq=10, gap=1, model_gen=None, with_fit=True,
             featnames=[]):
        """Load instance from saved state or create new if not exists"""
        return ModelStepper.load_utility(cls, folder=folder, name=name,
                                         lookback=lookback,
                                         minlookback=minlookback,
                                         fitfreq=fitfreq,
                                         gap=gap,
                                         model_gen=model_gen,
                                         with_fit=with_fit,
                                         featnames=featnames
                                         )

    def manage_history(self, dts, xseries, yserie, wgtserie, train_start_dt, train_stop_dt):

        keep_idx_1 = (self.last_dts >= train_start_dt) & (self.last_dts <= train_stop_dt)
        keep_idx_2 = (dts >= train_start_dt) & (dts <= train_stop_dt)
        new_dts = np.concatenate([
            self.last_dts[keep_idx_1],
            dts[keep_idx_2]
        ])
        new_yserie = np.concatenate([
            self.last_ymem[keep_idx_1],
            yserie[keep_idx_2]
        ])
        new_wserie = np.concatenate([
            self.last_wmem[keep_idx_1],
            wgtserie[keep_idx_2]
        ])
        if self.last_xmem.shape[0] == 0:
            nf = xseries.shape[1]
            self.last_xmem = np.ndarray(shape=(0, nf), dtype=np.float64)
        if self.last_xmem.shape[1] != xseries.shape[1]:
            pass
        new_xserie = np.concatenate([
            self.last_xmem[keep_idx_1],
            xseries[keep_idx_2]
        ])
        logger.info('ModelStepper :: updating the memory ')
        # assigning back
        self.last_dts = new_dts
        self.last_wmem = np.nan_to_num(new_wserie)
        self.last_ymem = np.nan_to_num(new_yserie)
        self.last_xmem = new_xserie

    def fit_model(self):
        model_loc = self.model_gen()
        model_loc.fit(X=self.last_xmem, y=self.last_ymem, sample_weight=self.last_wmem)

        featimp_lgbm(model_loc, self.featnames)
        return model_loc

    def update(self, dts, xseries, yserie=None, wgtserie=None):
        """
        We need to respect the dtsi here. 
        """
        if wgtserie is None:
            wgtserie = np.ones(xseries.shape[0])
        if yserie is None:
            assert not self.with_fit
            yserie = np.zeros(xseries.shape[0])
        self.validate_input(dts, dscode=np.ones(yserie.shape[0], dtype=np.int64),
                            serie=yserie, serie2=wgtserie)

        n = xseries.shape[0]
        first_time = dts[0]
        last_time = dts[-1]
        self.timeclf.update(dts, np.zeros(n), np.zeros(n))
        ltimes = self.timeclf.ltimes

        result = np.zeros(xseries.shape[0], dtype=np.float64)

        model_loc = None

        for ltime in ltimes:
            train_start_dt = ltime['train_start_dt']
            train_stop_dt = ltime['train_stop_dt']

            test_start_dt = ltime['test_start_dt']
            test_stop_dt = ltime['test_stop_dt']
            time_str = ltime['str']

            if test_stop_dt < first_time:
                continue
            if test_start_dt > last_time:
                break

            test_start_i = get_dts_max_before(dts, test_start_dt)
            test_stop_i = get_dts_max_before(dts, test_stop_dt)

            if self.with_fit:
                logger.info(f'Fitting model {time_str}')
                self.manage_history(dts, xseries, yserie, wgtserie, train_start_dt, train_stop_dt)
                model_loc = self.fit_model()
                self.hmodels += [{
                    'train_start_dt': train_start_dt,
                    'train_end_dt': train_stop_dt,
                    'model': model_loc}]
            else:
                model_loc = [x for x in self.hmodels if x['train_end_dt'] == train_stop_dt]['model']

            if model_loc is not None:
                n_feats = self.last_xmem.shape[1]
                ypred = model_loc.predict(xseries[test_start_i:test_stop_i, :n_feats])
                result[test_start_i:test_stop_i] = ypred
            else:
                result[test_start_i:test_stop_i] = 0.0

        return result
