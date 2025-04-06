

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

The beta between stocks is not 1.0

## Current issues

- add a conditioning on high rpt for the mual signal as well !

- finish test_model_stepper_sklearn_linear_changing_n



## We would need to have an optimizer as well
- not sure cvxpy is the best solution ?

## Ideas for the PfP
- we need a variable tick size that is function of the volatility !

## write pairs functions !
write incr_distance_corr stepper
puis write la selection des paires 
puis write la creation du nouveau rack !


## ML must use batches I think, no? 

## N-beats??? Attention matrix for stock tokens ??

## Adding Gaps / news related Gaps and so on!

## excess-volume * PIN ? 
## work with volume*minute return

## HMM forward pass for patterns


In the clustering version just add the correlation to the market ! as a feature for conditioning !


## work on the intraday volume curve ! and what is a correct excess volume

## TODO: add wgt in the pytorch models ! 

## Can we write the lightgbm as a batch learner ? 

coder le stochastic as a perform_sto


Priority 1 : The ML prediction shows some gaps :: it is almost fixed

Priority 2 : Deeply check the mual code it seems to have an issue in it!

## Trend R2 of the OLS vs droite !

# in each pair scale by the stdev first and then look at the spread.

## TODO: kbest put a warmup period please

# TODO: plot over time the fraction of volume that BTC and ETH represent !

## List of sorted things
The ML must be trained on forward_fh1_clip
P&L stats:- add sdt,edt and nb stocks

## TODO: add the day of the week

bucketplot against the time of the day please !!!



https://console.anthropic.com/settings/billing
https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview#install-and-authenticate
npm install -g @anthropic-ai/claude-code
https://console.anthropic.com/settings/billing

there is a folder stepper/ that contains some numba Stepper classes. They all have a method update that enables me    │
│   to backtest and quickly run in production some features that will be used to forecast stock/crypto prices. I then     │
│   created a folder stepperc/ inside I worked on incr_ewmkurt.pyx which is a cython version of stepper/incr_ewmkurt.py   │
│   ( numba) . So please read very carefully how I translated the numba code into a cython one. And please do the same    │
│   for all the other numba codes in stepper/ folder.  