
from utils.common import clean_folder
import numpy as np
import pandas as pd
import inspect
from .. import *



# pytest ./stepper/tests/test_save_load.py --pdb --maxfail=1

def test_reg_stepper_save_and_load(tmp_path):
    folder_name='test_stepper_save_load'
    
    stepper_classes = [cls for name, cls in globals().items() 
                   if isinstance(cls, type) and 'Stepper' in name]
    for stepper_class in stepper_classes:
        print(stepper_class)
        signature = inspect.signature(stepper_class.update)
        print(signature)
        nbparams=len(signature.parameters.keys())
        # I am expecting : ['self', 'dt', 'dscode', 'serie'])
        if nbparams>4:
            continue
        clean_folder(folder_name)
        clean_folder(folder_name+'2')

        #import pdb;pdb.set_trace()
        c = 5
        dt = np.array([1, 2, 3, 4, 5, 7, 8, 10, 11, 15, 20, 31], dtype=np.int64)
        dscode = np.array([1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
        serie1 = np.array([3.0, 100.0, 5.0, 9.0, 12.0, 5.0, 7.0, 8.0, 13.0, 15.0, 20.0, 110.0], dtype=np.float64)

        stepper1 = stepper_class.load(folder=folder_name,name='test')
        r1 = stepper1.update(dt[:c], dscode[:c], serie1[:c])
        stepper1.save()
        stepper2 = stepper_class.load(folder=folder_name,name='test')
        r2 = stepper2.update(dt[c:], dscode[c:], serie1[c:])

        stepper3 = stepper_class.load(folder=folder_name+'2',name='test')
        r3 = stepper3.update(dt, dscode, serie1)

        np.testing.assert_array_almost_equal(r3, np.concatenate([r1, r2]))
