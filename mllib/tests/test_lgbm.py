import lightgbm as lgb
import numpy as np

# pytest ./crptmidfreq/mllib/tests/test_lgbm.py --pdb --maxfail=1


def test_lgbm():

    # Generate synthetic dataset
    features = np.random.rand(1000, 10).astype(np.float32)  # 1000 samples, 10 features
    labels = np.random.rand(1000).astype(np.float32)         # 1000 labels

    train_set = lgb.Dataset(data=features, label=labels)
    model = lgb.train(params={}, train_set=train_set, num_boost_round=10)


def test_lgbm2():
    data_arr = np.random.rand(1000, 10).astype(np.float32)
    labels_arr = np.random.rand(1000).astype(np.float32)

    train_set = lgb.Dataset(
        data=data_arr,
        label=labels_arr,
    )

    # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
    lgb.train(
        train_set=train_set,
        params={},
        # params=model_param_generator(),
        # num_boost_round=100,
        # keep_training_booster=True,
    )
