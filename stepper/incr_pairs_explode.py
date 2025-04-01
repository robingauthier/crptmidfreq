
import numpy as np
from crptmidfreq.stepper.base_stepper import BaseStepper
from numba.typed import Dict
from numba.typed import List
from numba.core import types
from numba import njit


@njit
def explode_pairs_increment(dts, dscode, serie,
                            sdts, sdscode1, sdscode2,
                            rdts, rdscode1, rdscode2, rserie1, rserie2,
                            last_sdts, last_sdscode1, last_sdscode2):
    n = dts.shape[0]
    sn = sdts.shape[0]

    cdt = -1  # current dt
    fi = 0
    li = 0

    j = 0
    i = 0

    while i < n or j < sn:
        dt = dts[i]
        if j < sn and (sdts[j] < dts[i]):
            # update the memory of pairs
            if sdts[j] != last_sdts:
                # reset the memory
                last_sdts = sdts[j]
                last_sdscode1 = List.empty_list(types.int64)
                last_sdscode2 = List.empty_list(types.int64)
            last_sdscode1.append(sdscode1[j])
            last_sdscode2.append(sdscode2[j])
            j += 1

        if (dt > cdt) or (i == n-1):
            li = i
            dscode_loc = dscode[fi:li]
            serie_loc = serie[fi:li]

            # now creating pairs
            kn = len(last_sdscode1)
            for k in range(kn):
                dscode1 = last_sdscode1[k]
                dscode2 = last_sdscode2[k]

                rdts.append(cdt)
                rdscode1.append(dscode1)
                rdscode2.append(dscode2)
                val1_loc = serie_loc[dscode_loc == dscode1]
                if len(val1_loc) > 0:
                    rserie1.append(val1_loc[0])
                else:
                    rserie1.append(np.nan)
                val2_loc = serie_loc[dscode_loc == dscode2]
                if len(val2_loc) > 0:
                    rserie2.append(val2_loc[0])
                else:
                    rserie2.append(np.nan)
            fi = i
            cdt = dt

        i += 1


class PairsExplodeStepper(BaseStepper):
    """
    we will increase the size of the data and create pairs

    """

    def __init__(self, folder='', name=''):
        """
        """
        super().__init__(folder, name)
        # we store here the last list of pairs
        self.last_sdts = 0
        self.last_sdscode1 = List.empty_list(types.int64)
        self.last_sdscode2 = List.empty_list(types.int64)

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name):
        """Load instance from saved state or create new if not exists"""
        return PairsExplodeStepper.load_utility(cls, folder=folder, name=name)

    def update(self, dts, dscode, serie, sdts, sdscode1, sdscode2):
        """
        sdts  : selected pairs dts
        sdscode1 :selected pairs dscode1
        sdscode2 :selected pairs dscode2
        """
        rdts = List.empty_list(types.int64)
        rdscode1 = List.empty_list(types.int64)
        rdscode2 = List.empty_list(types.int64)
        rserie1 = List.empty_list(types.float64)
        rserie2 = List.empty_list(types.float64)
        explode_pairs_increment(dts, dscode, serie,
                                sdts, sdscode1, sdscode2,
                                rdts, rdscode1, rdscode2, rserie1, rserie2,
                                self.last_sdts, self.last_sdscode1, self.last_sdscode2)

        featd = {
            'dts': np.array(rdts, dtype=np.int64),
            'dscode1': np.array(rdscode1, dtype=np.float64),
            'dscode2': np.array(rdscode2, dtype=np.float64),
            'serie1': np.array(rserie1, dtype=np.float64),
            'serie2': np.array(rserie2, dtype=np.float64),
        }
        return featd
