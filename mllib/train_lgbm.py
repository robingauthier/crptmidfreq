import lightgbm as lgb
import numpy as np
import pandas as pd

from crptmidfreq.mllib.iterable_data_lgbm import ParquetIterableDataset


def featimp_lgbm(model_loc, feat_names):
    if model_loc is None:
        return
    fimp = pd.Series(model_loc.feature_importance(), index=model_loc.feature_name())
    print(fimp.sort_values(ascending=False).head(20))


def gen_lgbm_lin_params(n_samples=10e6):
    boosting_loc = 'gbdt'
    lgb_kwargs = dict(
        objective='regression_l2',
        verbosity=-1,
        n_jobs=10,
        linear_tree=False,  # main difference
        boosting=boosting_loc,
        learning_rate=1e-3,
        n_estimators=400,
        extra_trees=False,
        feature_fraction=0.5,
        bagging_fraction=0.8,
        bagging_freq=1 if boosting_loc == 'rf' else 5,
        max_bin=60,
        max_depth=100,  # was 6 then we decided to go for None
        num_leaves=31,  # trial.suggest_int('num_leaves', 3, 15), #
        lambda_l1=0.0,
        lambda_l2=0.0,  # not used
        drop_rate=0.1,  # dart
        min_data_in_leaf=max(200, int(0.05*n_samples)),
    )

    return lgb_kwargs


def train_model(folder_path,
                model_param_generator=gen_lgbm_lin_params,
                target='forward_fh1',
                filterfile=None,
                epochs=1,
                batch_size=128,
                num_boost_round=50,
                lr=1e-3,
                weight_decay=1e-3,
                batch_up=-1):
    """
    The goal here is to train a LightGBM model using a streaming dataset.
    """
    dataset = ParquetIterableDataset(folder_path, target=target, filterfile=filterfile)

    model = None
    feat_names = None
    cnt = 0
    for epoch in range(epochs):
        for (data, labels) in dataset:
            print(f'train_model_lgbm iter={cnt}')
            cnt += 1
            if feat_names is None:
                feat_names = data.columns.tolist()
            data_arr = data.values.astype(np.float32)
            labels_arr = labels.values.astype(np.float32)

            train_set = lgb.Dataset(
                data=data_arr,
                label=labels_arr,
                feature_name=data.columns.tolist(),
            )

            # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
            if model is None:
                model = lgb.train(
                    train_set=train_set,
                    params=model_param_generator(),
                    num_boost_round=num_boost_round,
                    keep_training_booster=True,
                )
            else:
                model = lgb.train(
                    train_set=train_set,
                    params=model_param_generator(),
                    num_boost_round=num_boost_round,
                    init_model=model,
                    keep_training_booster=True,
                )

    featimp_lgbm(model, feat_names)
    return model
