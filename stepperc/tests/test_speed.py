
from crptmidfreq.utils.common import clean_folder
from crptmidfreq.utils.log import get_logger
import numpy as np
import pandas as pd
import inspect
from crptmidfreq.stepper import *
logger = get_logger()
# pytest ./crptmidfreq/stepper/tests/test_speed.py --pdb --maxfail=1

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.expand_frame_repr', False)

def test_reg_stepper_save_and_load(tmp_path):
    folder_name = 'test_stepper_save_load'

    stepper_classes = [cls for name, cls in globals().items()
                       if isinstance(cls, type) and 'Stepper' in name]
    lr = []
    for stepper_class in stepper_classes:
        print(stepper_class)
        signature = inspect.signature(stepper_class.update)
        print(signature)
        nbparams = len(signature.parameters.keys())
        # I am expecting : ['self', 'dt', 'dscode', 'serie'])
        if nbparams > 4:
            continue
        clean_folder(folder_name)

        n = 1_000_000
        dt = np.arange(n, dtype=np.int64)
        dscode = np.int64(np.random.choice([1, 2, 3, 4], n))
        serie1 = np.random.normal(0, 1, n)

        stime = pd.to_datetime('now')
        try:
            stepper1 = stepper_class.load(folder=folder_name, name='test')
            r1 = stepper1.update(dt, dscode, serie1)
            logger.info(stepper1)
        except Exception as e:
            print(f"Error loading stepper: {e}")
            continue
        etime = pd.to_datetime('now')
        dtime = etime-stime
        lr += [{'stepper': str(stepper_class), 'dtime': dtime}]
        #assert dtime < pd.to_timedelta('0d 00:00:05'), f'the {str(stepper_class)} is too slow'
    rdf = pd.DataFrame(lr).sort_values('dtime', ascending=False)
    print(rdf)
    import pdb
    pdb.set_trace()
