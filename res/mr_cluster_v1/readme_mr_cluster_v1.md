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
