import os
import re
from functools import partial

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from crptmidfreq.mllib.train_lgbm import train_model as train_model_lgbm
from crptmidfreq.mllib.train_pytorch import train_model as train_model_torch
from crptmidfreq.stepper.base_stepper import BaseStepper
from crptmidfreq.stepper.incr_model_timeclf import (TimeClfStepper,
                                                    get_dts_max_before2)
from crptmidfreq.utils.common import get_logger


def keepimport():
    m = lgb.LGBMRegressor()


logger = get_logger()

# Does not use a lot of RAM. It relies on batches.


def filterfile(filename, sdt, edt):
    """
    Parse a filename like 'data_{fdts}_{ldts}.pq'.
    Return True if ldts < edt and ldts > sdt, else False.
    """
    pattern = r'^data_(\d+)_(\d+)\.pq$'
    match = re.match(pattern, filename)
    assert match is not None, 'issue'

    fdts_str, ldts_str = match.groups()
    fdts = int(fdts_str)
    ldts = int(ldts_str)

    if (ldts < edt) and (ldts > sdt):
        return True
    else:
        return False


class ModelBatchStepper(BaseStepper):
    """Relies on the pivot first and then runs an SVD"""

    def __init__(self, folder='', name='',
                 lookback=300,
                 ramlookback=10,
                 minlookback=100,
                 fitfreq=10,
                 gap=1,
                 epochs=10,
                 batch_size=128,
                 weight_decay=1e-3,
                 lr=1e-3,
                 model_gen=None,
                 is_torch=False,
                 with_fit=True,
                 featnames=[]):
        """
        """
        super().__init__(folder, name)
        self.folder_ml = self.folder+f'/ml_{name}/'

        # lgbm or pytorch model. .. both have different syntax
        self.is_torch = is_torch

        self.ramlookback = ramlookback

        self.timeclf = TimeClfStepper(folder=folder, name=name,
                                      lookback=lookback, minlookback=minlookback,
                                      fitfreq=fitfreq, gap=gap)

        self.timeclf_mem = TimeClfStepper(folder=folder, name=name,
                                          lookback=ramlookback,
                                          minlookback=ramlookback,
                                          fitfreq=ramlookback,
                                          gap=0)

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
        self.nfeats = 0

        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.lr = lr

        self.xcols = None

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, lookback=300, minlookback=100,
             fitfreq=10, gap=1,
             ramlookback=10,
             epochs=10,
             batch_size=128,
             weight_decay=1e-3,
             lr=1e-3,
             model_gen=None,
             is_torch=True,
             with_fit=True,
             featnames=[]):
        """Load instance from saved state or create new if not exists"""
        return ModelBatchStepper.load_utility(cls, folder=folder, name=name,
                                              lookback=lookback,
                                              ramlookback=ramlookback,
                                              minlookback=minlookback,
                                              fitfreq=fitfreq,
                                              gap=gap,
                                              weight_decay=weight_decay,
                                              model_gen=model_gen,
                                              with_fit=with_fit,
                                              is_torch=is_torch,
                                              featnames=featnames,
                                              epochs=epochs,
                                              batch_size=batch_size,
                                              lr=lr
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
        # assigning back
        self.last_dts = new_dts
        self.last_wmem = np.nan_to_num(new_wserie)
        self.last_ymem = np.nan_to_num(new_yserie)
        self.last_xmem = new_xserie

    def save_history(self):
        if self.last_xmem.shape[0] == 0:
            return
        df = pd.DataFrame(self.last_xmem)
        if self.xcols is not None:
            df.columns = self.xcols
        elif len(self.featnames) > 0:
            df.columns = self.featnames
        else:
            header = [f"feature_{j}" for j in range(df.shape[1])]
            df.columns = header

        df['forward_target'] = self.last_ymem
        #df['wgt'] = self.last_wmem
        fdts = np.min(self.last_dts)
        ldts = np.max(self.last_dts)
        os.makedirs(self.folder_ml, exist_ok=True)
        assert df.shape[0] > 0
        df = df.fillna(0.0)  # does not accept nan

        kurtdf = df.kurtosis()
        if kurtdf.max() > 40:
            print('Potential kurtosis issues')
            print(kurtdf[kurtdf > 20])

        df.to_parquet(self.folder_ml+f'data_{fdts}_{ldts}.pq')

        # resetting the memory
        self.last_xmem = np.ndarray(shape=(0, 0), dtype=np.float64)
        self.last_ymem = np.array([], dtype=np.float64)
        self.last_wmem = np.array([], dtype=np.float64)
        self.last_dts = np.array([], dtype=np.int64)

    def fit_model(self, sdt, edt):
        filterf = partial(filterfile, sdt=sdt, edt=edt)

        if self.is_torch:
            model_loc = self.model_gen(n_features=self.nfeats)
            model_loc = train_model_torch(self.folder_ml,
                                          model_loc,
                                          filterfile=filterf,
                                          target='forward_target',
                                          epochs=self.epochs,
                                          batch_size=self.batch_size,
                                          weight_decay=self.weight_decay,
                                          lr=self.lr,
                                          batch_up=-1)
        else:
            model_loc = train_model_lgbm(self.folder_ml,
                                         model_param_generator=self.model_gen,
                                         filterfile=filterf,
                                         target='forward_target',
                                         epochs=self.epochs,
                                         batch_size=self.batch_size,
                                         weight_decay=self.weight_decay,
                                         lr=self.lr,
                                         batch_up=-1)
        return model_loc

    def update(self, dts, xseries, yserie=None, wgtserie=None, xcols=None):
        """
        We need to respect the dtsi here. 
        """
        if self.nfeats == 0:
            self.nfeats = xseries.shape[1]

        if xcols is not None:
            self.xcols = xcols
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

        # Updating the timeclf
        self.timeclf.update(dts, np.zeros(n), np.zeros(n))
        self.timeclf_mem.update(dts, np.zeros(n), np.zeros(n))

        ltimes = self.timeclf.ltimes
        ltimes_mem = self.timeclf_mem.ltimes

        result = np.zeros(xseries.shape[0], dtype=np.float64)

        model_loc = None
        # saving down the data
        for ltime in ltimes_mem:
            train_start_dt = ltime['train_start_dt']
            train_stop_dt = ltime['train_stop_dt']

            test_start_dt = ltime['test_start_dt']
            test_stop_dt = ltime['test_stop_dt']
            time_str = ltime['str']

            if test_stop_dt < first_time:
                continue
            if test_start_dt > last_time:
                break

            test_start_i = get_dts_max_before2(dts, test_start_dt)
            test_stop_i = get_dts_max_before2(dts, test_stop_dt)

            self.manage_history(dts, xseries, yserie, wgtserie, train_start_dt, train_stop_dt)
            self.save_history()

        if len(ltimes) == 0:
            print('ModelBatch ::  caution ltimes is empty')

        # Now running fit and predict
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

            test_start_i = get_dts_max_before2(dts, test_start_dt)
            test_stop_i = get_dts_max_before2(dts, test_stop_dt)
            from pprint import pprint
            print('-'*20)
            pprint({
                'test_start_i': test_start_i,
                'test_stop_i': test_stop_i,
                'n': n,
            })
            pprint(ltime)

            if self.with_fit:
                logger.info(f'Fitting model {time_str}')
                model_loc = self.fit_model(sdt=train_start_dt, edt=train_stop_dt)
                self.hmodels += [{
                    'train_start_dt': train_start_dt,
                    'train_end_dt': train_stop_dt,
                    'model': model_loc}]
            else:
                model_loc = [x for x in self.hmodels if x['train_end_dt'] == train_stop_dt]['model']

            # to inspect the weights do:
            # print(model_loc.state_dict())
            #import pdb
            # pdb.set_trace()

            if model_loc is not None:
                if self.is_torch:
                    xseries_tensor = torch.tensor(xseries[test_start_i:test_stop_i], dtype=torch.float32)
                    ypred = model_loc.forward(xseries_tensor)
                    ypred = ypred.detach().numpy().flatten()
                else:
                    xseries_loc = xseries[test_start_i:test_stop_i].astype(np.float32)
                    ypred = model_loc.predict(xseries_loc)
                if np.std(ypred) < 1e-14:
                    # it happens when you have outliers in the input or you did not scale the inputs
                    assert False, 'ModelBatch issue your prediction is mainly 0.0'
                result[test_start_i:test_stop_i] = ypred
            else:
                result[test_start_i:test_stop_i] = 0.0

        #pct_zero = np.mean(result == 0.0)
        # if pct_zero > 0.2:
        #    import pdb
        #    pdb.set_trace()
        return result
