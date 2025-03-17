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
