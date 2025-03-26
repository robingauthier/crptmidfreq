

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

