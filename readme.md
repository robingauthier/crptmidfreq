

## General Structure
The general organisation of the repository is as follow:

1 - stepper folder contains the computation engine. Has what is an ewm, ...

2 - the layer on top is the featurelib/lib_v1.py
Each Stepper is wrapped into perform_ewm, perform_cs_rank ...

3- the layer on top is in strats these are pieces of code that are useful for strategies
and I intend to reuse

4- the research is in res/ and uses code above.

5- the prod folder is for what is intended to run in live trading. 



Now having said that I need to explain a couple additional folders:
- datahub : folder is for sourcing data. For instance minutely bars from binance
- mllib : contains wrappers for machine learning
- utils : as the name suggests, codes used a bit everywhere.



## Fee structure
At least you should put 2.0bps of costs
Clearly on this page below there used ot be a section for Futures and fees are a lot lower.
https://www.binance.com/en-GB/fee/trading

https://www.bitmex.com/app/wallet/fees
on bitmex we are at 1.5bps



## Current issues
 - the incr_pivot does not handle well additional stocks or stocks removed I think
 same for the clustering / Kmeans I fear.

 - the tret_xmkt does not have a daily mean of 0 I think ! --> FIXED NOW

- tret_csdemean y a un -1 qui apparait a la fin car toutes les valeurs sont 0.0 ? 

- you might want to clip before computing sret_kmeans_ewmstd100

- add a conditioning on high rpt for the mual signal as well !

- finish test_model_stepper_sklearn_linear_changing_n

