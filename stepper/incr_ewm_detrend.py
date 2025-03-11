import numpy as np
from stepper.incr_ewm import EwmStepper
from stepper.incr_ewm import update_ewm_values

class DetrendEwmStepper(EwmStepper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, dt, dscode, serie):
        """

        """
        # Input validation
        if not isinstance(dt, np.ndarray) or not isinstance(dscode, np.ndarray) or not isinstance(serie, np.ndarray):
            raise ValueError("All inputs must be numpy arrays")

        if len(dt) != len(dscode) or len(dt) != len(serie):
            raise ValueError("All inputs must have the same length")

        # Convert datetime64 to int64 nanoseconds for Numba
        timestamps = dt.astype('datetime64[ns]').astype('int64')

        # Update values and timestamps using numba function
        return serie-update_ewm_values(
            dscode, serie, timestamps,
            self.alpha, self.last_sum,self.last_wgt_sum, self.last_timestamps
        )
