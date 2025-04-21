import gc
import os

import numpy as np
import psutil

from crptmidfreq.utils.lazy_dict import \
    LazyDict  # <-- import your LazyDict class

# pytest ./crptmidfreq/utils/tests/test_lazy_dict.py --pdb --maxfail=1


def test_store_large_arrays():
    # 1) Create the LazyDict pointing to our temporary folder
    ld = LazyDict(folder='test_folder_dict', clean=True)

    # 2) Check memory usage before storing
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024**2)  # bytes

    # 3) Generate & store large arrays
    #    For demonstration, let's do 10 arrays of shape 2000x2000 => ~32MB each of float64
    num_arrays = 6
    arr_shape = 500_000_000

    for i in range(num_arrays):
        arr = np.float64(np.arange(arr_shape))  # big array in RAM
        ld[f"arr_{i}"] = arr               # stored on disk, freed from memory
        del arr                            # drop reference

    mem_after_store = process.memory_info().rss / (1024**2)
    overhead_mb = (mem_after_store - mem_before)

    # Allow some overhead because memory can fragment or spool temporarily
    assert overhead_mb < 50, (
        f"Memory usage grew by {overhead_mb:.2f} MB, which is unexpectedly high."
    )

    # 4) Retrieve the arrays (one by one) to verify correctness, then free them
    for i in range(num_arrays):
        arr_retrieved = ld[f"arr_{i}"]
        assert arr_retrieved.shape[0] == arr_shape
        # Minimal check to ensure data isn't garbage
        assert np.all(np.diff(arr_retrieved) > 0)
        del arr_retrieved
    gc.collect()

    mem_after_retrieval = process.memory_info().rss / (1024**2)
    overhead_after_mb = (mem_after_retrieval - mem_before)
    assert overhead_after_mb < 40, (
        f"Memory usage after retrieval is {overhead_after_mb:.2f} MB, still too high."
    )

    print(f"Memory before storing: {mem_before :.2f} MB")
    print(f"Memory after storing: {mem_after_store :.2f} MB")
    print(f"Memory after retrieval: {mem_after_retrieval :.2f} MB")

    # cleaning the folder in case!
    ld = LazyDict(folder='test_folder_dict', clean=True)
