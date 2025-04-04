import numpy as np
import pandas as pd


def generate_data(ndts=100, ndscode=100, rank=2):

    rng = np.random.RandomState(1234)

    # Rank=2 factors for each time (shape: 5 x 2)
    F = rng.normal(loc=0.0, scale=1.0, size=(ndts, rank))
    L = rng.normal(loc=0.0, scale=1.0, size=(ndscode, rank))
    vals = np.dot(F, np.transpose(L))

    df = pd.DataFrame(vals)
    pdf = df.unstack().reset_index()
    pdf = pdf.rename(columns={'level_0': 'dscode', 'level_1': 'dtsi', 0: 'serie_o'})
    pdf['univ'] = 1*(rng.uniform(size=pdf.shape[0]) < 0.7)
    pdf['serie_res'] = rng.normal(loc=0.0, scale=0.5, size=pdf.shape[0])
    pdf['serie'] = pdf['serie_o'] + pdf['serie_res']

    # we create missing points
    pdf = pdf.sample(frac=0.9, random_state=42)

    pdf = pdf.sort_values(['dtsi', 'dscode'])

    featd = {k: pdf[k].values for k in pdf.columns}
    return featd
