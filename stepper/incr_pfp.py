
import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict
from crptmidfreq.stepper.base_stepper import BaseStepper


@njit
def calculate_restype(restype, price, entrylevel, maxlevel, minlevel, direction, nan):
    """
    Calculate the value for a specific restype.
    """
    if restype == 1:
        return price
    elif restype == 2:
        return (maxlevel / entrylevel - 1) if direction > 0 else (maxlevel / entrylevel - 1)
    elif restype == 3:
        return (maxlevel / minlevel - 1) if direction > 0 else -(minlevel / maxlevel - 1)
    elif restype == 4:
        return direction
    elif restype == 5:
        return maxlevel
    elif restype == 6:
        return entrylevel
    elif restype == 7:
        return maxlevel if direction > 0 else nan
    elif restype == 8:
        return nan if direction > 0 else maxlevel
    return nan


@njit
def update_pfp_values(dts, dscodes, prices, nbrev, last_pfp_states):
    """
    Incremental update of PFP values for each dscode.
    """

    # pfp rounded price
    res_price = np.empty(len(prices), dtype=np.float64)
    # direction
    res_dir = np.empty(len(prices), dtype=np.float64)
    # entry level
    res_el = np.empty(len(prices), dtype=np.float64)
    # performance since entry
    res_perf = np.empty(len(prices), dtype=np.float64)
    res_perf2 = np.empty(len(prices), dtype=np.float64)
    # duration in state
    res_dur = np.empty(len(prices), dtype=np.float64)

    for i in range(len(prices)):
        code = dscodes[i]
        price = prices[i]
        ts = dts[i]

        # Initialize or fetch the last state for this dscode
        if code not in last_pfp_states:
            last_pfp_states[code] = Dict.empty(types.unicode_type, types.float64)
            last_pfp_states[code]["direction"] = 1.0
            last_pfp_states[code]["entrytime"] = ts
            last_pfp_states[code]["entrylevel"] = price
            last_pfp_states[code]["minlevel"] = price
            last_pfp_states[code]["maxlevel"] = price
        state = last_pfp_states[code]
        entrytime = state["entrytime"]
        direction = state["direction"]
        entrylevel = state["entrylevel"]
        minlevel = state["minlevel"]
        maxlevel = state["maxlevel"]

        if direction > 0:
            if price > maxlevel:
                maxlevel = price
            if price <= maxlevel - nbrev:
                direction = -1
                entrylevel = price
                minlevel = price
                maxlevel = price
                entrytime = ts
        else:  # direction < 0
            if price < minlevel:
                minlevel = price
            if price >= minlevel + nbrev:
                direction = 1
                entrylevel = price
                minlevel = price
                maxlevel = price
                entrytime = ts

        # Update the state for this dscode
        state["direction"] = direction
        state["entrylevel"] = entrylevel
        state["entrytime"] = entrytime
        state["minlevel"] = minlevel
        state["maxlevel"] = maxlevel
        last_pfp_states[code] = state
        res_price[i] = price
        # direction
        res_dir[i] = direction
        # entry level
        res_el[i] = entrylevel
        if entrylevel != 0:
            # performance since entry
            res_perf[i] = price / entrylevel - 1
        else:
            res_perf[i] = 0.0
        res_perf2[i] = price - entrylevel
        # duration in state
        res_dur[i] = ts - entrytime
    return res_price, res_dir, res_el, res_perf, res_perf2, res_dur


class PfPStepper(BaseStepper):
    def __init__(self, folder='', name='', nbrev=3, tick=1e-4):
        super().__init__(folder, name)
        self.nbrev = nbrev
        self.tick = tick

        # Initialize empty state
        self.last_pfp_states = Dict.empty(
            key_type=types.int64,
            value_type=types.DictType(
                types.unicode_type, types.float64
            )
        )

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name,  nbrev=3, tick=1e-4):
        """Load instance from saved state or create new if not exists"""
        return PfPStepper.load_utility(cls,
                                       folder=folder,
                                       name=name,
                                       nbrev=nbrev,
                                       tick=tick)

    def update(self, dts, dscodes, prices):
        """
        Update PFP values for each code and return the PFP values for all restypes.

        Args:
            prices: numpy array of prices.
            dscodes: numpy array of categorical codes.

        Returns:
            Dict mapping each restype to numpy arrays of results.
        """
        self.validate_input(dts, dscodes, prices)

        # code does not work with non nan prices
        assert np.sum(np.isnan(prices)) == 0
        prices_int = (prices // self.tick)
        res_price, res_dir, res_el, res_perf, res_perf2, res_dur = update_pfp_values(
            dts.view(np.int64), dscodes, prices_int,
            self.nbrev, self.last_pfp_states
        )
        return res_price * self.tick, res_dir, res_el * self.tick, res_perf, res_perf2, res_dur
