

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

## HMM forward pass for patterns



## P&L stats:
- add sdt,edt and nb stocks
