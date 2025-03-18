import numpy as np
from crptmidfreq.stepper.incr_ewm import EwmStepper
from crptmidfreq.stepper.incr_ewm import update_ewm_values

class DetrendEwmStepper(EwmStepper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, dt, dscode, serie):
        """

        """
        self.validate_input(dt,dscode,serie)
        # Update values and timestamps using numba function
        return serie-update_ewm_values(
            dscode, serie, dt.view(np.int64),
            self.alpha, self.last_sum,self.last_wgt_sum, self.last_timestamps
        )
