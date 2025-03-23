

def _reconstruct_TDigest(compression):
    #from crptmidfreq.stepper.tdigest.tdigestloc import TDigest
    #return TDigest(compression)
    #print('calling _reconstruct_TDigest')
    from crptmidfreq.stepper.tdigest.tdigestloc import TDigest
    return TDigest(compression)