
import numpy as np
from numba import njit
from numba.core import types
from numba.typed import Dict

from crptmidfreq.stepper.base_stepper import BaseStepper


@njit(cache=True)
def collapse_pairs_increment(dts, dscode1, dscode2, serie,
                             rpserie, rpcnt,
                             ):
    last_dt = dts[0]
    n = dts.shape[0]
    j = 0
    for i in range(n):
        dt = dts[i]
        code1 = dscode1[i]
        code2 = dscode2[i]
        val = serie[i]

        if dt > last_dt:
            j += 1
            last_dt = dt

        rpserie[code1][j] = rpserie[code1][j]+val
        rpserie[code2][j] = rpserie[code2][j]-val

        rpcnt[code1][j] = rpcnt[code1][j]+1
        rpcnt[code2][j] = rpcnt[code2][j]+1


class PairsCollapseStepper(BaseStepper):
    """
    we go back to the dscode
    """

    def __init__(self, folder='', name=''):
        """
        """
        super().__init__(folder, name)
        self.last_dscodes = None

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name):
        """Load instance from saved state or create new if not exists"""
        return PairsCollapseStepper.load_utility(cls, folder=folder, name=name)

    def update(self, dts, dscode1, dscode2, serie):
        """
        sdts  : selected pairs dts
        sdscode1 :selected pairs dscode1
        sdscode2 :selected pairs dscode2
        """
        # nan are causing issues
        serie = np.float64(np.nan_to_num(serie))
        dscode1 = np.int64(np.nan_to_num(dscode1, nan=-1))
        dscode2 = np.int64(np.nan_to_num(dscode2, nan=-1))

        udts = np.sort(np.unique(dts))
        n = udts.shape[0]
        udscode = np.unique(np.concatenate([dscode1, dscode2]))
        if self.last_dscodes is None:
            self.last_dscodes = udscode

        rpsum = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.float64, 1, 'C')
        )
        rpcnt = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.float64, 1, 'C')
        )
        rpserie = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.float64, 1, 'C')
        )

        for code in udscode:
            rpsum[code] = np.zeros(n, dtype=np.float64)
            rpcnt[code] = np.zeros(n, dtype=np.float64)
            rpserie[code] = np.zeros(n, dtype=np.float64)

        # dts, dscode1, dscode2, serie,rpserie, rpcnt,
        collapse_pairs_increment(dts, dscode1, dscode2, serie,
                                 rpsum, rpcnt)
        for code in udscode:
            rpserie[code] = np.where(rpcnt[code] > 0, rpsum[code]/rpcnt[code], 0.0)

        return udts, rpserie
