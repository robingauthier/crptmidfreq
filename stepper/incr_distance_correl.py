
import numpy as np
from crptmidfreq.stepper.base_stepper import BaseStepper
from numba.typed import Dict
from numba.typed import List
from numba.core import types
from numba import njit


@njit
def incremental_distance_correl(
    dts,
    xseriesd,
    clusterd,
    last_xmem,
    rdts,
    rdscode1,
    rdscode2,
    rclus,
    rdist,
        lookback,
        mempos,
        warmpos,
        fitfreq,
        last_i
):
    """
    Incremental pivot function that organizes data by unique timestamps and 
    assigns each dscode as a separate column without using pandas.

    X = U D V.T


     X = U D Vt
     Vt is fixed
     D is fixed
     U is going to change

    """
    n = 0
    for code in xseriesd:
        n = max(n, xseriesd[code].shape[0])

    # adding a space for new dscodes
    for code in xseriesd:
        if code not in last_xmem:
            last_xmem[code] = np.zeros(lookback, dtype=np.float64)

    for i in range(n):
        dt = dts[i]
        mempos = (mempos+1) % lookback  # position in last_mem . It is circular
        warmpos += 1  # min nb of points to compute svd
        last_i += 1
        # updating values
        for code in xseriesd:
            last_xmem[code][mempos] = xseriesd[code][i]

        if warmpos < lookback:
            continue

        recompute_cond = last_i >= fitfreq
        if recompute_cond:
            last_i = 0
            for code1 in xseriesd:
                for code2 in xseriesd:
                    if code1 >= code2:
                        continue
                    clus1 = np.int64(clusterd[code1][i])
                    clus2 = np.int64(clusterd[code2][i])
                    if clus1 != clus2:
                        continue
                    corr = np.corrcoef(
                        np.nan_to_num(last_xmem[code1]),
                        np.nan_to_num(last_xmem[code2])
                    )
                    rdts.append(dt)
                    rdscode1.append(code1)
                    rdscode2.append(code2)
                    rdist.append(corr[0, 1])
                    rclus.append(clus1)


class CorrelDistanceStepper(BaseStepper):
    """Relies on the pivot first and then runs an SVD"""

    def __init__(self, folder='', name='', lookback=300, fitfreq=10):
        """
        """
        super().__init__(folder, name)
        self.fitfreq = fitfreq
        self.last_i = 0
        self.lookback = lookback

        # Initialize empty state
        self.mempos = 0
        self.warmpos = 0

        # Memory
        self.last_xmem = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.float64, 1, 'C')
        )

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, lookback=300, fitfreq=10):
        """Load instance from saved state or create new if not exists"""
        return CorrelDistanceStepper.load_utility(cls, folder=folder, name=name,
                                                  lookback=lookback, fitfreq=fitfreq)

    def update(self, dts, xseriesd, clusterd=None):
        """
        pserie : pivoted serie
        puniv : pivoted PIT universe with 1/0

        Returns:
        resultd : numba Dict dscode: array of residuals
        result_D : 2 dim array with the diagonal values of the matrix D
        """
        n = dts.shape[0]
        if clusterd is None:
            clusterd = Dict.empty(
                key_type=types.int64,  # dsocode
                value_type=types.Array(types.int64, 1, 'C'),  # 1/0 for currently in or out univ
            )
            for code in xseriesd:
                clusterd[code] = np.ones(n, dtype=np.int64)

        rdscode1 = List.empty_list(types.int64)
        rdscode2 = List.empty_list(types.int64)
        rclus = List.empty_list(types.int64)
        rdist = List.empty_list(types.float64)
        rdts = List.empty_list(types.float64)

        incremental_distance_correl(
            dts, xseriesd, clusterd,
            self.last_xmem,
            rdts,
            rdscode1,
            rdscode2,
            rclus,
            rdist,
            self.lookback,
            self.mempos,
            self.warmpos,
            self.fitfreq,
            self.last_i,
        )
        rfeat = {
            'dtsi': np.array(rdts, dtype=np.int64),
            'dscode1': np.array(rdscode1, dtype=np.int64),
            'dscode2': np.array(rdscode2, dtype=np.int64),
            'dist': np.array(rdist, dtype=np.float64),
        }
        return rfeat
