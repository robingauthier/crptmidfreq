import numpy as np
import pandas as pd

from crptmidfreq.mllib.lgbm_sklearn import LGBMModel
from crptmidfreq.mllib.lgbm_sklearn_optuna import LGBMModelOptuna
from crptmidfreq.mllib.timeclf import TimeSplitClf
from crptmidfreq.season.res.kagstore.kagstore_data import get_data
from crptmidfreq.season.res.kagstore.kagstore_feature import add_features
from crptmidfreq.season.res.kagstore.kagstore_feature import add_features_lag
from crptmidfreq.utils.common import to_csv

#from crptmidfreq.season.gluonts_v1 import fit_gluonts_model
# sales_log= log(1+sales)
# exp(sales_log)= 1+sales
# sales= exp(sales_log)-1

data_start = pd.to_datetime('2013-01-01')
valid_start = pd.to_datetime('2015-09-01')  # for optuna
test_start = pd.to_datetime('2016-09-01')
data_end = pd.to_datetime('2017-08-15')


def evaluate_model(df, name='', verbose=True):
    first_date = df['date'].min()
    mae = (df['sales']-df['pred_sales']).abs().mean()
    nbpoints = df.shape[0]
    if verbose:
        print(f'MAE model {name} is mae:{mae:.2f}  -- cnt:{nbpoints} -- date>={first_date}')
    return mae


def dump_data(df, name='', with_wcols=True):
    wcols = ['date', 'sales', 'pred_sales']
    fams = ['LIQUOR,WINE,BEER', 'AUTOMOTIVE', 'FROZEN FOODS']
    for fam in fams:
        if with_wcols:
            tdf = df.loc[lambda x:x['family'] == fam][wcols].sort_values('date')
        else:
            tdf = df.loc[lambda x:x['family'] == fam].sort_values('date')
        tdf['d'] = tdf['sales']-tdf['pred_sales']
        tdf['cd'] = tdf['d'].cumsum()
        to_csv(tdf, f'example_kagstore_{name}_{fam}')


def main_lgbm_optuna():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=False)

    df, categorical = LGBMModelOptuna.preprocess_data(df, f2['categorical'])
    f2['categorical'] = categorical

    df_train = df.loc[lambda x: x['date'] <= test_start].copy()
    df_test = df.loc[lambda x: x['date'] > test_start].copy()

    model = LGBMModelOptuna(cat_features=f2['categorical'], num_features=f2['numerical'], n_trials=30)
    model.fit(df_train, target_col='sales', wgt_col='one')
    df_test['pred_sales'] = model.predict(df_test)
    df['pred_sales'] = model.predict(df)

    mae = evaluate_model(df_test, name='kaggle-store', verbose=True)
    dump_data(df, name='lgbm_optuna')
    return {'mae': mae, 'name': 'lgbm_optuna'}


def main_lgbm():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=False)

    df, categorical = LGBMModelOptuna.preprocess_data(df, f2['categorical'])
    f2['categorical'] = categorical

    df_train = df.loc[lambda x: x['date'] <= test_start].copy()
    df_test = df.loc[lambda x: x['date'] > test_start].copy()

    model = LGBMModel(cat_features=f2['categorical'], num_features=f2['numerical'])
    model.fit(df_train, target_col='sales', wgt_col='one')
    df_test['pred_sales'] = model.predict(df_test)
    df['pred_sales'] = model.predict(df)

    mae = evaluate_model(df_test, name='kaggle-store', verbose=True)
    dump_data(df, name='lgbm')
    return {'mae': mae, 'name': 'lgbm'}


def main_lgbm_by_familly():
    from crptmidfreq.mllib.local import LocalModel2
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=False)

    df, categorical = LGBMModelOptuna.preprocess_data(df, f2['categorical'])
    f2['categorical'] = categorical

    df_train = df.loc[lambda x: x['date'] <= test_start].copy()
    df_test = df.loc[lambda x: x['date'] > test_start].copy()

    def model_gen():
        model = LGBMModel(cat_features=f2['categorical'], num_features=f2['numerical'])
        return model
    model = LocalModel2(model_generator=model_gen, by_col='family')
    model.fit(df_train, target_col='sales', wgt_col='one')
    df_test['pred_sales'] = model.predict(df_test)
    df['pred_sales'] = model.predict(df)

    mae = evaluate_model(df_test, name='kaggle-store', verbose=True)
    dump_data(df, name='lgbm_family')
    return {'mae': mae, 'name': 'lgbm_family'}


def main_lgbm_rolling():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=False)
    df['one'] = 1.0
    df, categorical = LGBMModelOptuna.preprocess_data(df, f2['categorical'])
    f2['categorical'] = categorical

    df['dscode_old'] = df['dscode']
    df = df.set_index(['dscode', 'date'])
    df = df.sort_index(level='date')
    df = df.rename(columns={'dscode_old': 'dscode'})

    def simple_model():
        model = LGBMModel(cat_features=f2['categorical'], num_features=f2['numerical'])
        return model

    model = TimeSplitClf(
        fct=simple_model,
        # Split arguments
        train_freq=250,
        min_train_size=400,
        max_train_days=600,
        train_gap=1,
        train_remove_covid=True,
        fit_colname_syntax=True,
        n_jobs=0,
    )
    ypred = model.fit_predict(df, df['sales'], df['one'])
    df['pred_sales'] = ypred

    # we need to remove the index now
    df = df.drop(['dscode'], axis=1)
    df = df.reset_index()

    df_test = df.loc[lambda x: x['date'] > test_start].copy()
    mae = evaluate_model(df_test, name='kaggle-store', verbose=True)
    dump_data(df, name='lgbm_rolling')
    return {'mae': mae, 'name': 'lgbm_rolling'}


def main_lgbm_relative():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=True)

    df, categorical = LGBMModelOptuna.preprocess_data(df, f2['categorical'])
    f2['categorical'] = categorical

    df_train = df.loc[lambda x: x['date'] <= test_start].copy()
    df_test = df.loc[lambda x: x['date'] > test_start].copy()

    model = LGBMModelOptuna(cat_features=f2['categorical'], num_features=f2['numerical'], n_trials=30)
    model.fit(df_train, target_col='sales_vs_ewm', wgt_col='one')
    df_test['pred_sales_vs_ewm'] = model.predict(df_test)
    df_test['pred_sales_log'] = df_test['pred_sales_vs_ewm']+df_test['wgt']
    df_test['pred_sales'] = np.exp(df_test['pred_sales_log'])-1.0

    mae = evaluate_model(df_test, name='kaggle-store', verbose=True)
    return {'mae': mae, 'name': 'lgbm_relative'}


def main_lgbm_bottom_up():
    df, f1 = get_data(agglevel=None)
    df, f2 = add_features(df, f1, use_log=False)

    df, categorical = LGBMModelOptuna.preprocess_data(df, f2['categorical'])
    f2['categorical'] = categorical

    df_train = df.loc[lambda x: x['date'] <= test_start].copy()

    # dataset is too big for optuna
    model = LGBMModel(cat_features=f2['categorical'], num_features=f2['numerical'])
    model.fit(df_train, target_col='sales', wgt_col='one')
    df['pred_sales'] = model.predict(df)

    # Aggregation now
    dfg = df.groupby(['family', 'date'])\
        .agg({'sales': 'sum', 'pred_sales': 'sum', 'onpromotion': 'sum'}).reset_index()

    dfg_test = dfg.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(dfg_test, name='kaggle-store')
    dump_data(df, name='lgbm_bottom_up')
    return {'mae': mae, 'name': 'lgbm_bottom_up'}


def main_naive_ewm():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=False)
    df['pred_sales'] = df['wgt']

    df_test = df.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(df_test, name='kaggle-store')
    dump_data(df, name='naivecst')

    return {'mae': mae, 'name': 'naive_cst'}


def main_naive_1y_lag():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=True)
    # we clip the YoY rate
    df['sales_vs_ewm_lag1y'] = df['sales_vs_ewm_lag1y'].clip(lower=-0.5, upper=0.5)
    df['pred_sales_log'] = df['sales_vs_ewm_lag1y']+df['wgt']
    df['pred_sales'] = np.exp(df['pred_sales_log'])-1.0

    df_test = df.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(df_test, name='kaggle-store')
    dump_data(df, name='naive1y', with_wcols=False)
    return {'mae': mae, 'name': 'naive_1y_lag'}


def main_naive_1y_lag_hols():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=True, use_hols=True)
    # we clip the YoY rate
    df['sales_vs_ewm_lag1y'] = df['sales_vs_ewm_lag1y'].clip(lower=-0.5, upper=0.5)
    df['pred_sales_log'] = df['sales_vs_ewm_lag1y']+df['wgt']
    df['pred_sales'] = np.exp(df['pred_sales_log'])-1.0

    df_test = df.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(df_test, name='kaggle-store')
    dump_data(df, name='naive1yh')

    return {'mae': mae, 'name': 'naive_1y_lag_hols'}


def main_naive_1y_lag_absolute():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=False)
    df['pred_sales'] = df['sales_lag1y']

    df_test = df.loc[lambda x:x['date'] >= test_start]
    mae = evaluate_model(df_test, name='kaggle-store')
    dump_data(df, name='naive1y_lag_absolute')
    return {'mae': mae, 'name': 'naive_1y_lag_absolute'}


def main_feedforward():
    from crptmidfreq.mllib.feedforward_sklearn import FeedForwardRegressor
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=False)

    df, categorical = LGBMModelOptuna.preprocess_data(df, f2['categorical'])
    f2['categorical'] = categorical

    df_train = df.loc[lambda x: x['date'] <= test_start].copy()

    feats = f2['numerical']+f2['categorical']
    model = FeedForwardRegressor(input_dim=len(feats))
    model.fit(df_train[feats].fillna(0.0), df_train['sales'].fillna(0.0))
    df['pred_sales'] = model.predict(df[feats].fillna(0.0))

    df_test = df.loc[lambda x:x['date'] > test_start]
    mae = evaluate_model(df_test, name='kaggle-store', verbose=True)
    dump_data(df, name='feedforward')
    return {'mae': mae, 'name': 'feedforward'}


def main_nbeats():
    from crptmidfreq.mllib.nbeats_sklearn import NBeatsNet
    df, f1 = get_data()
    df, f2 = add_features_lag(df, f1, use_log=False)

    df_train = df.loc[lambda x: x['date'] <= test_start].copy()

    feats = f2['numerical']+f2['categorical']
    model = NBeatsNet(
        stack_types=('trend', 'seasonality', 'generic'),
        thetas_dim=(3, 3, 3),  # must have same length as stack_types, <=4
        backcast_length=len(feats),
        forecast_length=1,
        hidden_layer_units=3,
        nb_harmonics=2,
        learning_rate=1e-4,
        nb_blocks_per_stack=1,
        weight_decay=1e-2,
    )
    model.compile(loss='mse', optimizer='adam')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {total_params}")

    model.fit(df_train[feats].fillna(0.0), df_train['sales'].fillna(0.0), batch_size=32, epochs=30)
    df['pred_sales'] = model.predict(df[feats].fillna(0.0))

    df_test = df.loc[lambda x:x['date'] > test_start]
    mae = evaluate_model(df_test, name='kaggle-store', verbose=True)
    dump_data(df, name='nbeats')
    return {'mae': mae, 'name': 'nbeats'}


def main_nbeats2():
    from crptmidfreq.mllib.nbeats2_sklearn import NBeatsNet
    df, f1 = get_data()
    df, f2 = add_features_lag(df, f1, use_log=False)

    df_train = df.loc[lambda x: x['date'] <= test_start].copy()

    feats = f2['numerical']+f2['categorical']
    model = NBeatsNet(
        input_size=len(feats),
        trend_layer_size=20,
        seasonality_layer_size=20,
        generic_layer_size=20,
        generic_blocks=2,
    )

    model.fit(df_train[feats].fillna(0.0), df_train['sales'].fillna(0.0))

    print(model)
    ee = model.get_params()['module']()
    total_params = sum(p.numel() for p in ee.parameters() if p.requires_grad)
    print(f"Total trainable params: {total_params}")

    df['pred_sales'] = model.predict(df[feats].fillna(0.0))

    df_test = df.loc[lambda x:x['date'] > test_start]
    mae = evaluate_model(df_test, name='kaggle-store', verbose=True)
    dump_data(df, name='nbeats')
    return {'mae': mae, 'name': 'nbeats'}


def main_lgbm_relativeyoy():
    df, f1 = get_data()
    df, f2 = add_features(df, f1, use_log=True)

    df, categorical = LGBMModelOptuna.preprocess_data(df, f2['categorical'])
    f2['categorical'] = categorical

    df_train = df.loc[lambda x: x['date'] <= test_start].copy()
    df_test = df.loc[lambda x: x['date'] > test_start].copy()

    model = LGBMModelOptuna(cat_features=f2['categorical'], num_features=f2['numerical'], n_trials=30)
    model.fit(df_train, target_col='sales_log_yoy', wgt_col='one')
    df_test['pred_sales_log_yoy'] = model.predict(df_test)
    df_test['pred_sales_log'] = df_test['pred_sales_log_yoy']+df_test['sales_log_lag1y']
    df_test['pred_sales'] = np.exp(df_test['pred_sales_log'])-1.0

    mae = evaluate_model(df_test, name='kaggle-store', verbose=True)
    fam = 'LIQUOR,WINE,BEER'
    to_csv(df.loc[lambda x:x['family'] == fam], 'tt')
    return {'mae': mae, 'name': 'lgbm_relativeyoy'}


def main():
    """
                mae                   name
    2    2064.309754         lgbm_bottom_up
    0    2340.353410                   lgbm
    12   2513.668320       lgbm_relativeyoy
    9    2622.141601            lgbm_family
    4    2669.612973           naive_1y_lag
    5    2678.844947      naive_1y_lag_hols
    1    2781.346278            lgbm_optuna
    8    2901.365811           lgbm_rolling
    10   3002.353171            feedforward
    6    3616.366427  naive_1y_lag_absolute
    3    4298.212521          lgbm_relative
    7    5058.685658              naive_cst
    11  19455.578270                 nbeats


    """
    r = []
    r += [main_lgbm()]
    r += [main_lgbm_optuna()]
    r += [main_lgbm_bottom_up()]  # slow
    r += [main_lgbm_relative()]
    r += [main_naive_1y_lag()]
    r += [main_naive_1y_lag_hols()]
    r += [main_naive_1y_lag_absolute()]
    r += [main_naive_ewm()]
    r += [main_lgbm_rolling()]
    r += [main_lgbm_by_familly()]
    r += [main_feedforward()]
    r += [main_nbeats()]
    r += [main_lgbm_relativeyoy()]

    rdf = pd.DataFrame(r)
    print('-'*20)
    print('-'*20)
    print(rdf.sort_values('mae'))


# ipython -i -m crptmidfreq.season.res.kagstore.kagstore_main
if __name__ == '__main__':
    # TODO add 1 model per category pls !
    # main_lgbm_relative()
    # main_naive_1y_lag()
    # main_lgbm_bottom_up()
    # main_feedforward()
    # main_lgbm_relativeyoy()
    main_nbeats2()
    # main_lgbm_bottom_up()
