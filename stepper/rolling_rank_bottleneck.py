import numpy as np
from bottleneck import move_rank
from crptmidfreq.stepper.base_stepper import BaseStepper


class BottleneckRankStepper(BaseStepper):
    """
    Ignores the dscode to be a lot faster ! 
    """

    def __init__(self, folder='', name='', window=1000):
        """

        """
        super().__init__(folder, name)
        self.window = window
        self.last_values = np.array([], dtype=np.float64)

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, window=1000):
        """Load instance from saved state or create new if not exists"""
        return BottleneckRankStepper.load_utility(cls, folder=folder, name=name, window=window)

    def update(self, dt, dscode, values):
        assert np.all(np.diff(dt.view(np.int64)) >= 0)
        assert self.window > 0
        self.validate_input(dt, dscode, values)
        if self.last_values.shape[0] > 0:
            nvalues = np.concatenate([self.last_values, values], axis=0)
            nrm = self.last_values.shape[0]
        else:
            nvalues = values
            nrm = 0
        nres = move_rank(nvalues, self.window)
        self.last_values = nvalues[-self.window:]
        res = nres[nrm:]
        assert res.shape[0] == values.shape[0]
        return res
