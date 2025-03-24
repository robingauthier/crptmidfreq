from lightgbm import LGBMRegressor

# https://lightgbm.readthedocs.io/en/latest/Parameters.html


def gen_lgbm_lin_v1(n_samples):

    boosting_loc = 'gbdt'

    lgb_kwargs = dict(
        objective='regression_l2',
        verbosity=-1,
        n_jobs=10,
        linear_tree=True,  # main difference
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
    return LGBMRegressor(**lgb_kwargs)
