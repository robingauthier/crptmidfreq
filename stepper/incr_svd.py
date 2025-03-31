
import numpy as np
from crptmidfreq.stepper.base_stepper import BaseStepper
from numba.typed import Dict
from numba.typed import List
from numba.core import types
from numba import njit
# because fitfreq can be a lot higher ( 1 or 2) we will not use TimeClf here


@njit
def stack(last_xmem, last_univ, lookback):
    # pX = np.stack([last_xmem[k] for k in last_univ])
    # pX = np.transpose(pX)
    # pX = np.nan_to_num(pX)
    nuniv = len(last_univ)
    pX = np.zeros((lookback, nuniv), dtype=np.float64)
    # Fill it in a loop
    for j in range(nuniv):
        code = last_univ[j]
        pX[:, j] = np.nan_to_num(last_xmem[code])
    return np.ascontiguousarray(pX)


@njit
def diag(arr):
    n = len(arr)
    res = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        res[i, i] = arr[i]
    return np.ascontiguousarray(res)


@njit
def incremental_svd(xseriesd, univd,
                    last_xmem, last_univ, last_xorder,
                    lookback,
                    mempos, warmpos, fitfreq, n_comp,
                    result_resid):
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

    for code in xseriesd:
        result_resid[code] = np.zeros(n, dtype=np.float64)  # residuals of pseries
    for code in last_xmem:
        result_resid[code] = np.zeros(n, dtype=np.float64)  # residuals of pseries
    result_D = np.zeros((n, n_comp), dtype=np.float64)  # how much each factor explains risk

    # adding a space for new dscodes
    for code in xseriesd:
        if code not in last_xmem:
            last_xmem[code] = np.zeros(lookback, dtype=np.float64)
            last_xorder = np.append(last_xorder, code)

    last_Vt = None
    last_Dinv = None
    D_trunc = None
    Vt_trunc = None

    n_comp_loc = n_comp

    for i in range(n):
        mempos = (mempos+1) % lookback  # position in last_mem . It is circular
        warmpos += 1  # min nb of points to compute svd

        # updating values
        for code in last_xorder:
            if code in xseriesd:
                last_xmem[code][mempos] = xseriesd[code][i]
            else:
                last_xmem[code][mempos] = 0.0

        if warmpos < lookback:
            continue

        recompute_cond = (last_Vt is None or i % fitfreq == 0)

        if recompute_cond:
            last_univ = np.array([k for k in last_xorder if univd[k][i] > 0], dtype=np.int64)
            pX = stack(last_xmem, last_univ, lookback)
            U, D, Vt = np.linalg.svd(pX, full_matrices=False)
            is_sorted_desc = np.all(np.diff(D) <= 0)
            assert is_sorted_desc, f'issue SVD sorting for i={i}'
            last_U = U
            last_D = D
            last_Vt = Vt

            last_Dinv = np.where(last_D > 0, 1/last_D, 0)
            last_Dinv = np.ascontiguousarray(last_Dinv)
            last_V = np.ascontiguousarray(np.transpose(Vt))

            D_trunc = np.ascontiguousarray(last_D[:n_comp_loc])
            Vt_trunc = np.ascontiguousarray(last_Vt[:n_comp_loc, :])
        else:
            # X = U D Vt   D and Vt  will not change
            # U = X V D-1
            pX = stack(last_xmem, last_univ, lookback)
            last_U = pX @ last_V @ diag(last_Dinv)
            last_D = D
            last_Vt = Vt

        # computing the truncated X
        n_min = min(last_D.shape[0], last_univ.shape[0])
        n_min = min(n_comp, n_min)
        U_trunc = np.ascontiguousarray(last_U[:, :n_comp_loc])

        # computing an approximation
        X_ap = U_trunc @ diag(D_trunc)@Vt_trunc
        X_resid = pX - X_ap

        # Setting the results
        for jj in range(len(last_univ)):
            code = last_univ[jj]
            if univd[code][i] > 0:
                result_resid[code][i] = X_resid[mempos, jj]
        result_D[i, :n_min] = last_D[:n_min]

    # residuals, % variance explained, Factor returns
    return result_resid, result_D


class SVDStepper(BaseStepper):
    """Relies on the pivot first and then runs an SVD"""

    def __init__(self, folder='', name='', lookback=300, fitfreq=10, n_comp=2):
        """
        """
        super().__init__(folder, name)
        self.n_comp = n_comp
        self.fitfreq = fitfreq
        self.lookback = lookback

        # Initialize empty state
        self.mempos = 0
        self.warmpos = 0

        self.last_univ = np.array([], dtype=np.int64)
        self.last_xorder = np.array([], dtype=np.int64)

        self.last_xmem = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.float64, 1, 'C')
        )
        self.last_umem = Dict.empty(
            key_type=types.int64,  # dsocode
            value_type=types.float64,  # 1/0 for currently in or out univ
        )

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, lookback=300, fitfreq=10, n_comp=2):
        """Load instance from saved state or create new if not exists"""
        return SVDStepper.load_utility(cls, folder=folder, name=name,
                                       lookback=lookback, fitfreq=fitfreq, n_comp=n_comp)

    def update(self, dts, xseriesd, univd=None):
        """
        pserie : pivoted serie
        puniv : pivoted PIT universe with 1/0

        Returns:
        resultd : numba Dict dscode: array of residuals
        result_D : 2 dim array with the diagonal values of the matrix D
        """
        n = dts.shape[0]
        if univd is None:
            univd = Dict.empty(
                key_type=types.int64,  # dsocode
                value_type=types.Array(types.float64, 1, 'C'),  # 1/0 for currently in or out univ
            )
            for code in xseriesd:
                univd[code] = np.ones(n, dtype=np.float64)
        else:
            for code in univd:
                univd[code] = np.nan_to_num(univd[code])
                univd[code] = np.int64(univd[code])

        resultd = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.float64, 1, 'C')
        )
        resultd, result_D = incremental_svd(
            xseriesd, univd,
            self.last_xmem,
            self.last_univ,
            self.last_xorder,
            self.lookback,
            self.mempos,
            self.warmpos,
            self.fitfreq,
            self.n_comp,
            resultd)
        return resultd, result_D
