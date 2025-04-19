
import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict

from crptmidfreq.stepper.base_stepper import BaseStepper


def get_alpha(window=1.0):
    """Convert half-life to alpha"""
    if window==0:
        return 0.0
    return 1 - np.exp(np.log(0.5) / window)


@njit(cache=True)
def update_holt_winters(codes, values, timestamps,
                        alpha, beta, gamma,
                        mem,
                        time_unit,
                        use_seasonality,
                        level_dict, trend_dict, season_dict, last_timestamps):
    result = np.empty(len(codes), dtype=np.float64)
    nmem=2*mem
    
    for i in range(len(codes)):
        code = codes[i]
        value = values[i]
        ts = timestamps[i]

        if np.isnan(value):
            result[i] = np.nan
            continue

        last_ts = last_timestamps.get(code, np.int64(0))
        if ts < last_ts:
            raise ValueError("Timestamps must be strictly increasing per code")

        last_timestamps[code] = ts

        level = level_dict.get(code, value)
        trend = trend_dict.get(code, 0.0) 

        if code not in season_dict:
            season_dict[code] = np.ones(nmem, dtype=np.float64)
        if use_seasonality:
            pos_season = np.int64(np.mod(np.int64(ts/time_unit ),mem))
            pseason = season_dict.get(code)[pos_season]
        else:
            pseason = 1.0
            pos_season=0

        # https://otexts.com/fpp2/holt-winters.html
        #St = α(Xt/It−p) + (1 − α)(St−1 + Tt−1), (6)
        #Tt = β(St − St−1) + (1 − β)Tt−1, (7)
        #It = γ(Xt/St) + (1 − γ)It−p, (8)
        #Xˆt(h) = (St + hTt)It−p+h,

        if np.isnan(level):
            # Initialize components
            level = value / pseason
            trend = 0.0
        else:
            prev_level = level
            level = alpha * (value / pseason) + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
            if use_seasonality:
                # in theory we adjust for season but p times ago ... hence ignore that !
                season = gamma * (value / level) + (1 - gamma) * pseason
                # clipping season
                season = min(season,20.0)
                season = max(season,0.05)
            else:
                season=1.0

        level_dict[code] = level
        trend_dict[code] = trend
        season_dict[code][pos_season]=season

        # 1-step ahead forecast
        forecast = (level + trend) * season
        result[i] = forecast

    return result


class HoltWinterStepper(BaseStepper):

    def __init__(self, folder='', name='',    alpha=10.0, beta=10.0, gamma=10.0,seasonality=100,time_unit=1,
                 use_seasonality=True):
        super().__init__(folder, name)
        self.use_seasonality = use_seasonality
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.seasonality=seasonality
        self.time_unit=time_unit

        # Initialize empty state
        self.level_dict = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        self.trend_dict = Dict.empty(
            key_type=types.int64,
            value_type=types.float64
        )
        self.season_dict = Dict.empty(
            key_type=types.int64,
            value_type=types.Array(types.float64, 1, 'C')
        )
        self.last_timestamps = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name,  alpha=10.0, beta=10.0, gamma=10.0,seasonality=100,time_unit=1,use_seasonality=True):
        """Load instance from saved state or create new if not exists"""


        return HoltWinterStepper.load_utility(cls, folder=folder, name=name, 
                                              alpha=alpha, 
                                              beta=beta, 
                                              gamma=gamma,
                                              time_unit=time_unit,
                                              seasonality=seasonality,
                                              use_seasonality=use_seasonality)

    def update(self, dt, dscode, serie):
        """
        Update EWM values for each code and return the EWM values for each input row

        Args:
            dt: numpy array of datetime64 values
            dscode: numpy array of categorical codes
            serie: numpy array of values to process

        Returns:
            numpy array of same length as input arrays containing EWM values
        """
        self.validate_input(dt, dscode, serie)

        # Update values and timestamps using numba function
        res = update_holt_winters(dscode, serie, dt.view(np.int64),
                        get_alpha(self.alpha), 
                        get_alpha(self.beta), 
                        get_alpha(self.gamma),
                        self.seasonality,
                        self.time_unit,
                        self.use_seasonality,
                        self.level_dict, 
                        self.trend_dict, 
                        self.season_dict, 
                        self.last_timestamps)
        return res
