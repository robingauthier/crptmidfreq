do Cmd+Shift+V to get a preview



MLPWork/data_enriched/mr/mr_cluster_v1.py

Task 1: check how the size of the universe looks like over time


Details 

- Create a pit_univ flag. I was merging asof with this universe. This way it was entirely fixed.

- n_cluster is a pct of nb stocks

- I had multiple transformations of my returns before the clustering : log, qnorm,rank.
- Clusdf was a table I would merge as of with

- My clustering could be fuzzy

- I would weight inside each cluster by prev_lots

Once clustering is done : compute a zscore
- step1 : I remove a long term trend
- step2 : P - ewm(P)
- step3 : normalize by a sum of std_5,std_10,std_20,std_40



- Work on log returns in order 
- Do remove the long term trend
- windows = [4,8,15]
    

- final signal is weighed by wgt = prev_lots

TODO:bucketplot of volume traded / market cap vs P&L yep
perform a rolling kmeans


I have some path features with history:
pdf = path_simple(df['zs'], levels=[0.8, 1.5], buffer_value_mult=0.2, buffer_value_additive=0.1, hist=3)


Things specific to crypto to add:
- time since token is listed -- there are weirds things going on a start of token listing


# TODO: model the intraday volume curve

# Pure time serie operators we need to code:
- PfP is important
- path operators
- Use the SVD
- Macd opeators are important
- volume macd are important 
- volume*ret ? 


-- We will then need to implement pairs operators :: we need to start thinking about that yes !


## Distribution of an Ewm(X)

Let $ x_1, x_2, \ldots, x_n \sim \mathcal{N}(0, \sigma^2) $ be an i.i.d. sequence of Gaussian random variables.

Define the Exponentially Weighted Moving Average (EWMA) recursively as:
$$
\text{ewm}_x(n+1) = \alpha \cdot \text{ewm}_x(n) + (1 - \alpha) \cdot x_n
$$
with $ 0 < \alpha < 1 $, and assume $ \text{ewm}_x(1) = 0 $ (or some fixed value).

We can unroll the recursion:
$$
\text{ewm}_x(n) = (1 - \alpha) \sum_{k=0}^{n-1} \alpha^k x_{n-1-k}
$$

As \( n \to \infty \), this becomes a weighted sum of i.i.d. Gaussian variables:
$$
y = (1 - \alpha) \sum_{k=0}^{\infty} \alpha^k x_k
$$

Since each \( x_k \sim \mathcal{N}(0, \sigma^2) \), and the sum is linear, \( y \sim \mathcal{N}(0, \text{Var}[y]) \), where:
$$
\text{Var}[y] = (1 - \alpha)^2 \sum_{k=0}^{\infty} \alpha^{2k} \cdot \sigma^2
= \sigma^2 (1 - \alpha)^2 \sum_{k=0}^{\infty} \alpha^{2k}
$$

The geometric sum gives:
$$
\sum_{k=0}^\infty \alpha^{2k} = \frac{1}{1 - \alpha^2}
$$

So:
$$
\text{Var}[y] = \sigma^2 \cdot \frac{(1 - \alpha)^2}{1 - \alpha^2} = \sigma^2 \cdot \frac{1 - \alpha}{1 + \alpha}
$$

\textbf{Therefore, the limiting distribution of the EWMA is:}
$$
\boxed{
\text{ewm}_x(n) \sim \mathcal{N}\left(0,\; \sigma^2 \cdot \frac{1 - \alpha}{1 + \alpha}\right)
}
$$

Hence code must be:
```
        # ewm(X) has Var = Var(x_i)* (1-alpha)/(1+alpha)
        featd[f'todel_{ewm_col}'] = featd[ewm_col]*np.sqrt((1+alpha)/(1-alpha))
```


### How many points does 4 years represent ?
we have 2Million dates in the data.
```
In [9]: 60*24*365*4/1e6
Out[9]: 2.1024
``` 




        # this one below is good
        featd = main(icfg={
            'forward_xmkt': True,
            'kmeans_or_svd_or_naive': 'mkt',
            'hardcoded_universe': True,
            'nb_fetures': 2,
        })


                                                  col   abs_sel       sel         sr          rpt           dd          ddn          hold    abssel
61                                     sigf_timeofday  0.996161 -0.996161 -18.626635    -8.505123    29.202835     0.334442    356.436290  0.996161
48                      sigf_sret_ewmstd_macdr100x500  0.996107 -0.996107 -24.118539    -9.442281    33.965746     0.223372    347.444840  0.996107
37                 sigf_sret_log1_cumsum_pfp1x3_perf2  0.960089  0.959726  14.386651    25.042347     0.407202    54.961470    610.092687  0.959726
72               sigf_sret_log1_cumsum_pfp0.5x3_perf2  0.960089  0.959726  12.239376    17.672603     0.473986    47.850528    495.366970  0.959726
36                         sigf_mual_ddpct_appops1000  0.947667 -0.935087  -7.502512    -4.233010    46.574049     0.981541    120.920261  0.935087
34                    sigf_sret_ewmkurt800_appops1000  0.486243  0.479795   7.696063     1.916272     0.735968    48.363878     52.855626  0.479795
13                       sigf_sret_ewmstd_macdr20x100  0.470972 -0.470972 -25.071414    -2.063894    35.988263     0.193674     72.104656  0.470972
73                         sigf_gap_1000x1x2.0_ewm100  0.431302 -0.415274  -3.186567    -1.458652    30.507674    16.051560     79.543012  0.415274
