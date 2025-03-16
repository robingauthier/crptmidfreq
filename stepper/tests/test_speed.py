
from utils.common import clean_folder
import numpy as np
import pandas as pd
import inspect
from .. import *



# pytest ./stepper/tests/test_speed.py --pdb --maxfail=1

def test_reg_stepper_save_and_load(tmp_path):
    folder_name='test_stepper_save_load'
    
    stepper_classes = [cls for name, cls in globals().items() 
                   if isinstance(cls, type) and 'Stepper' in name]
    lr=[]
    for stepper_class in stepper_classes:
        print(stepper_class)
        signature = inspect.signature(stepper_class.update)
        print(signature)
        nbparams=len(signature.parameters.keys())
        # I am expecting : ['self', 'dt', 'dscode', 'serie'])
        if nbparams>4:
            continue
        clean_folder(folder_name)

        n=40000
        dt = np.arange(n, dtype=np.int64)
        dscode = np.int64(np.random.choice([1,2,3,4],n))
        serie1 = np.random.normal(0,1,n)


        stime= pd.to_datetime('now')
        stepper1 = stepper_class.load(folder=folder_name,name='test')
        r1 = stepper1.update(dt, dscode, serie1)
        etime= pd.to_datetime('now')
        dtime=etime-stime
        lr+=[{'stepper':str(stepper_class),'dtime':dtime}]
        assert  dtime<pd.to_timedelta('0d 00:00:05'),f'the {str(stepper_class)} is too slow'
    rdf=pd.DataFrame(lr).sort_values('dtime',ascending=False)
    print(rdf)
    import pdb;pdb.set_trace()
