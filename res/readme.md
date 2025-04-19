

excess volume vs yesterday at the same time / you want some intraday volume profile
svd and clustering bien sur
ce serait bien d'avoir un excel avec le secteur de chaque coin



- The very first strategy should simply perform a KMeans 
- The second one should perform a SVD

- the third ones will be pairs based

- we will need a detailled fondamental mapping of tokens

Portmanteau tests for linearity of stationary time series

## Causality
https://microprediction.medium.com/a-new-way-to-detect-causality-in-time-series-interview-with-alejandro-rodriguez-dominguez-9257e8783d7f



## State Space S4 models
https://github.com/state-spaces/s4?tab=readme-ov-file#getting-started-with-s4


## Changing the target/forward 
- L1 piece wise trends as target
- Proba to be in quantile 1,2,3,4 or 5 forward looking like the M6 competition
(multi-class classification problem)


## Clustering
OPTICS or Kmeans

## Graph Neural Networks  GNN
https://medium.com/stanford-cs224w/temporal-graph-learning-for-stock-prediction-58429696f482
https://distill.pub/2021/gnn-intro/

https://drpress.org/ojs/index.php/HSET/article/view/6649/6444

The input data are stock price series. The VG module is first applied to transform the raw market price series into corresponding time series graphs, in order to deal with the evolving and chaotic property of stock price series. Next, the GNN module takes the converted graphs as input, trains, learns, and makes predictions based upon them. 

L1 piecewice approximation gives a better forward fh1 than anything else !
https://medium.com/stanford-cs224w/temporal-graph-learning-for-stock-prediction-58429696f482


https://www.lgresearch.ai/publication/view?seq=51
Transformer on graphs where each node is a token !


## Modele N-Beats
https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/nbeats.py
Darts
Philippe Remi
ServiceNow NBeats
https://nixtlaverse.nixtla.io/neuralforecast/docs/getting-started/quickstart.html
https://github.com/Nixtla/transfer-learning-time-series?tab=readme-ov-file  --- link to pre-trained models
https://github.com/unit8co/darts/blob/master/examples/14-transfer-learning.ipynb


## LGBM of one stock return vs others as features.

## Q learning for pairs trading
https://link.springer.com/article/10.1007/s11227-021-04013-x



For datasets
https://featuretools.alteryx.com/en/stable/resources/resources_index.html


## ICA instead of PCA
https://www.northinfo.com/documents/404.pdf
https://github.com/ronnieqt/ICA-in-Finance/blob/master/Independent%20Component%20Analysis%20for%20Financial%20Time%20Series.pdf



## Lot of interesting things we already know on
https://goldinlocks.github.io/Machine-Learning-for-Trading/


## Finetuning timeserie models
https://huggingface.co/amazon/chronos-bolt-small
https://huggingface.co/models?pipeline_tag=time-series-forecasting&p=1&sort=downloads

N-Beats must be trained on real life timeseries and fine tuned on financial data
 using pretrained N-BEATS on M4 series.

https://unit8.com/resources/transfer-learning-for-time-series-forecasting/  -- use the M4 competition to pre-train N-Beats
https://www.kaggle.com/competitions/m5-forecasting-accuracy/data


https://ts.gluon.ai/stable/
https://github.com/philipperemy/n-beats
https://github.com/ServiceNow/N-BEATS/tree/master/common
https://forecastingdata.org/ --- data for timeserie prediction
https://github.com/Nixtla/nixtla/tree/main
https://github.com/awslabs/gluonts/blob/dev/src/gluonts/torch/model/simple_feedforward/estimator.py

https://www.amazon.science/code-and-datasets/spectro-riemannian-graph-neural-networks

https://arxiv.org/html/2410.16032v1


# Graph based for MR portfolios
https://tevgeniou.github.io/EquityRiskFactors/bibliography/MeanRevertingPortfolios.pdf

# good resource
https://github.com/adamd1985/quant_research/blob/main/oscilators-quant.ipynb



## pair features:
Rolling Correlation:

Compute the correlation coefficient between the two stocks over a rolling window. A dynamic correlation can indicate periods when the stocks are more or less in sync.

Hedge Ratio / Beta:

Estimate the hedge ratio by regressing one stock’s price on the other’s. This tells you how many units of one stock are needed to hedge against movements in the other.

Cointegration Metrics:

Test for cointegration (using methods like the Engle-Granger two-step approach) to see if a stable long-term relationship exists. The cointegration coefficient itself can be an important feature.

Volatility Features:

Calculate the rolling volatility of each stock and the spread. Changes in volatility can signal shifts in market dynamics or risk.

Derivative Features (Momentum and Curvature):

First Derivative (Momentum): Measure the rate of change of the spread to capture momentum.

Second Derivative (Curvature): Assess the acceleration or deceleration in the spread, which can help identify inflection points.

Cross-Correlation with Lags:

Analyze the cross-correlation function with various lags to detect any lead-lag relationships, which might hint that one stock tends to move before the other.

## M6 competition paper
https://www.sciencedirect.com/science/article/pii/S0169207024001079#:~:text=The%20M6%20forecasting%20competition%20requested,their%20forecasting%20and%20investment%20performance.


Paper of the winning M4 competition
https://www.sciencedirect.com/science/article/abs/pii/S0169207019300895
The objective of our meta-learning approach is to derive a set of weights for combining the forecasts generated from a pool of methods. (Feature-based FORecast Model Averaging) means we add features to the model that averages.


For several decades, simple time series forecasting methods like exponential smoothing (Hyndman et al., 2002) and Theta (Spiliotis et al., 2020) have been outperforming relatively more sophisticated approaches.

What is Theta ? https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#theta
https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html#mstl


Saisonality seems to be important !!!




## ATA method !
Hanife Taylan Selamlar, who won 2nd prize in the investment challenge (110th position in the forecasting challenge), used the ATA method, which resembles models of the exponential smoothing family, to forecast (on a monthly basis) the daily frequency of each asset’s percentage returns across the five quintiles

https://dergipark.org.tr/en/download/article-file/858239
https://cran.r-project.org/web/packages/ATAforecasting/ATAforecasting.pdf

Was clearly one of the best of the M3 competition the The Ata
https://atamethod.wordpress.com/wp-content/uploads/2016/11/presentation.pdf

$$

S_t \;=\; \frac{p}{t} \, X_t \;+\; \frac{t-p}{t} \,\bigl(S_{t-1} + T_{t-1}\bigr),\\

T_t \;=\; \frac{q}{t}\,\bigl(S_t - S_{t-1}\bigr)\;+\; \frac{t-q}{t}\,T_{t-1},\\

\hat{X}_t(h) \;=\; S_t \;+\; h\,T_t,\\
\text{where } t > p \ge q,\\
S_t = X_t \quad \text{for } t \le p,\\
T_t = X_t - X_{t-1} \quad \text{for } t \le q,\\
T_{1} = 0,\\
p \in \{1,2,\dots,n\},\quad q \in \{0,1,\dots,n-1\},\quad p \ge q.\\
$$

The same exists for the multiplicative trend model 
$$
S_{t} \;=\; \frac{p}{t}\,X_{t}
\;+\; \frac{t - p}{t}\,\bigl(S_{t-1} \cdot T_{t-1}\bigr),\\
T_{t} \;=\; \frac{q}{t}\,\biggl(\frac{S_{t}}{S_{t-1}}\biggr)
\;+\; \frac{t - q}{t}\,T_{t-1},\\
\hat{X}_{t}(h) \;=\; S_{t}\,\bigl(T_{t}\bigr)^{h}.
$$



## Theta method
For several decades, simple time series forecasting methods like exponential smoothing (Hyndman et al., 2002) and Theta (Spiliotis et al., 2020) have been outperforming relatively more sophisticated approaches, such as neural networks and other computational intelligence methods. 

https://www.sciencedirect.com/science/article/abs/pii/S0377221720300242

The Theta method, proposed by Assimakopoulos and Nikolopoulos (2000), is a univariate forecasting method which decomposes the original data into two or more lines, called Theta lines, extrapolates them using forecasting models of our choice, and then combines their predictions to obtain the final forecasts.
https://www.statsmodels.org/dev/examples/notebooks/generated/theta-model.html


It often yields robust, accurate forecasts without needing overly complicated modeling steps.

## ARIMA
The ATA Method is a new alternative forecasting method. This method is alternative to two major forecasting approaches: Exponential Smoothing and ARIMA

## EWM/ Holt trend / Holt Winter
Simple exponential smoothing (Brown, 1959)
$$
St = αXt + (1 − α)(St−1), (1)
$$

Xˆt(h) = St , (2)


Holt’s additive trend method (Holt,1957)
$$
St = αXt + (1 − α)(St−1 + Tt−1), (3)
Tt = β(St − St−1) + (1 − β)Tt−1, (4)
Xˆt(h) = St + hTt
, (5)
$$


Holt-Winters exponential smoothing method (Winters, 1960)
$$
St = α(Xt/It−p) + (1 − α)(St−1 + Tt−1), (6)
Tt = β(St − St−1) + (1 − β)Tt−1, (7)
It = γ(Xt/St) + (1 − γ)It−p, (8)
Xˆt(h) = (St + hTt)It−p+h,
$$


## Markov Transition Field
https://pyts.readthedocs.io/en/stable/modules/image.html
MarkovTransitionField discretizes a time series into bins. It then computes the Markov Transition Matrix of the discretized time series. Finally it spreads out the transition matrix to a field in order to reduce the loss of temporal information.

## WEASEL
https://pyts.readthedocs.io/en/stable/modules/transformation.html#weasel
WEASEL extracts words with several sliding windows of different sizes, and selects the most discriminative words according to the chi-squared test

## TBATS / BATS

Handles multiple seasonality periods (common in daily data that show weekly and annual seasonal patterns).
More complex than standard ES or ARIMA but can be excellent in certain domains (e.g., forecasting daily web traffic).

## Good account : https://github.com/FranklinMa810?page=3&tab=repositories



https://github.com/FranklinMa810/modern-cpp-tutorial
https://github.com/FranklinMa810/modern-cpp-tutorial/blob/master/book/en-us/01-intro.md


https://github.com/FranklinMa810/feature-engineering-book



## Microsturture ECP:
https://github.com/fiquant/marketsimulator

## Site de l'X quantreg.com


https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2


https://www.datasciencewithmarco.com/blog
