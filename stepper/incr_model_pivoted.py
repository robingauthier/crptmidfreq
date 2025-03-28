
import numpy as np
from crptmidfreq.stepper.base_stepper import BaseStepper
from crptmidfreq.utils.common import get_logger
from numba.typed import Dict
from numba.core import types
from crptmidfreq.stepper.incr_model_timeclf import get_dts_max_before
from crptmidfreq.stepper.incr_model_timeclf import TimeClfStepper

# Quite different from incr_model because of the way the Kmeans works
# stocks are features.. but the nb of features can vary because I can add new stocks.
# This is the fundamental problem we are trying to solve here

logger = get_logger()


class PivotModelStepper(BaseStepper):
    """Relies on the pivot first and then runs an SVD or Kmeans"""

    def __init__(self, folder='', name='',
                 lookback=300,
                 minlookback=100,
                 fitfreq=10,
                 gap=1,
                 model_gen=None,
                 with_fit=True,
                 is_kmeans=False):
        """
        """
        super().__init__(folder, name)
        self.fitfreq = fitfreq
        self.gap = gap
        self.lookback = lookback
        self.minlookback = minlookback
        self.model_gen = model_gen  # you need to call model_gen() to create a new model
        self.with_fit = with_fit
        self.is_kmeans = is_kmeans

        # Initialize empty state
        self.last_xmem = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.float64, 1, 'C')
        )
        self.last_dts = np.array([], dtype=np.int64)
        self.last_ymem = np.array([], dtype=np.float64)
        self.last_wmem = np.array([], dtype=np.float64)
        self.last_xorder = []  # order of dscodes

        self.model_i = 0  # counting the models

        # This is very special to the Kmeans
        self.kmeans_cat = None

        # History of models
        self.hmodels = []

        self.timeclf = TimeClfStepper(folder=folder, name=name,
                                      lookback=lookback, minlookback=minlookback,
                                      fitfreq=fitfreq, gap=gap)

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, lookback=300, minlookback=100,
             fitfreq=10, gap=1, model_gen=None, with_fit=True, is_kmeans=False):
        """Load instance from saved state or create new if not exists"""
        return PivotModelStepper.load_utility(cls, folder=folder, name=name,
                                              lookback=lookback, minlookback=minlookback,
                                              fitfreq=fitfreq, gap=gap,
                                              model_gen=model_gen, with_fit=with_fit,
                                              is_kmeans=is_kmeans
                                              )

    def manage_history(self, dts, xseriesd, yserie, wgtserie, train_start_dt, train_stop_dt):

        keep_idx_1 = (self.last_dts >= train_start_dt) & (self.last_dts <= train_stop_dt)
        keep_idx_2 = (dts >= train_start_dt) & (dts <= train_stop_dt)
        new_dts = np.concatenate([
            self.last_dts[keep_idx_1],
            dts[keep_idx_2]
        ])
        self.last_dts = new_dts
        if yserie is not None:
            new_yserie = np.concatenate([
                self.last_ymem[keep_idx_1],
                yserie[keep_idx_2]
            ])
            self.last_ymem = np.nan_to_num(new_yserie)
        if wgtserie is not None:
            new_wserie = np.concatenate([
                self.last_wmem[keep_idx_1],
                wgtserie[keep_idx_2]
            ])
            self.last_wmem = np.nan_to_num(new_wserie)
        for k in xseriesd.keys():
            self.last_xmem[k] = np.concatenate([
                self.last_xmem[k][keep_idx_1],
                xseriesd[k][keep_idx_2]
            ])
        logger.info('PivotedModelStepper :: updating the memory ')

        # filling nan
        for k in xseriesd.keys():
            self.last_xmem[k] = np.nan_to_num(self.last_xmem[k])

    def fit_model(self):
        logger.info(f'Fitting Kmeans {self.folder} {self.name} i={self.model_i}')
        model_loc = self.model_gen()
        pX = np.array([self.last_xmem[k] for k in self.last_xorder])
        # model_loc.fit_predict(X=np.transpose(pX))
        # Kmeans works on X =(n_samples, n_features)
        # and returns labels = (n_samples) -- Index of the cluster each sample belongs to.
        model_loc.fit_predict(X=pX)
        if self.is_kmeans:
            assert model_loc.labels_.shape[0] == len(self.last_xorder)
            self.kmeans_cat = model_loc.labels_
        self.model_i += 1
        return model_loc

    def get_pX(self, xseriesd):
        """ensures the order is always the same ! """
        pX = np.array([xseriesd[k] for k in self.last_xorder])
        return np.transpose(pX)

    def normalize_new_dscode(self, xseriesd):
        # making sure every dscode has an xmem and is in series
        # this is taking care of new symbols..
        for code in xseriesd:
            if code not in self.last_xmem:
                self.last_xmem[code] = np.zeros(self.lookback, dtype=np.float64)
                self.last_xorder += [code]
        for code in self.last_xmem:
            if code not in xseriesd:
                xseriesd[code] = np.zeros(n, dtype=np.float64)
        assert np.all(
            sorted(list(xseriesd))
            ==
            sorted(list(self.last_xmem)))
        return xseriesd

    def update(self, dts, xseriesd, yserie=None, wgtserie=None):
        """
        xseriesd must be a pivoted table numba Dict where keys are dscode
        For example it can be obtained using
        pdts, pfeatd = perform_pivot(featd=featd,
                                 feats=[incol],
                                 folder=folder,
                                 name=name,
                                 r=r)
        xseriesd = pfeatd

        This function will return result a pivoted table numba Dict where keys are dscode
        and ndts the dates.

        """

        assert isinstance(xseriesd, Dict)
        n = dts.shape[0]
        for k, v in xseriesd.items():
            assert len(v) == n, 'issue on key={k}'

        result = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.float64, 1, 'C')
        )
        ndts = np.array([], dtype=np.int64)

        # Updating the timeclf
        first_time = dts[0]
        last_time = dts[-1]
        self.timeclf.update(dts, np.zeros(n), np.zeros(n))
        ltimes = self.timeclf.ltimes

        xseriesd = self.normalize_new_dscode(xseriesd)

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
                self.manage_history(dts, xseriesd, yserie, wgtserie, train_start_dt, train_stop_dt)
                model_loc = self.fit_model()
                self.hmodels += [{
                    'train_start_dt': train_start_dt,
                    'train_end_dt': train_stop_dt,
                    'model': model_loc}]
            else:
                model_loc = [x for x in self.hmodels if x['train_end_dt'] == train_stop_dt]['model']

            for code in self.last_xorder:
                if not code in result:
                    result[code] = np.zeros(ndts.shape[0], dtype=np.float64)

            # Prediction part
            if model_loc is not None:
                if not self.is_kmeans:
                    pX = self.get_pX(xseriesd)
                    ypred = model_loc.predict(pX[test_start_i:test_stop_i, :])
                    # TODO: modify below. it is wrong
                    for k in range(test_start_i, test_stop_i):
                        result[k] = ypred
                else:
                    ndts = np.append(ndts, test_start_dt)
                    for i in range(len(self.last_xorder)):
                        code = self.last_xorder[i]
                        if i < len(self.kmeans_cat):
                            result[code] = np.append(result[code], self.kmeans_cat[i])
                        else:
                            result[code] = np.append(result[code], -1)

        # check all have same lenght
        n2 = ndts.shape[0]
        for code in result:
            assert result[code].shape[0] == n2

        return result, ndts
