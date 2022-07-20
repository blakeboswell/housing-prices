import sys
sys.path.append('..')

from preprocess.load import load_data
from preprocess.baseline import baseline_trans
from metrics import rmsle
from shared_train import run_experiment

from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer


params = {
    'model__min_child_weight': [1, 5, 8, 10],
    'model__gamma': [0.5, 1, 1.5, 2, 5],
    'model__subsample': [0.4, 0.6, 0.8, 1.0],
    'model__colsample_bytree': [0.4, 0.6, 0.8, 1.0],
    'model__max_depth': [3, 4, 5, 8, 10]
}

rmsle_score = make_scorer(rmsle, greater_is_better=False)

pipe = Pipeline(
    steps=[
        ('trans', baseline_trans),
        ('model', XGBRegressor(n_estimators=600, tree_method='gpu_hist', gpu_id=0, random_state=0))
    ]
)

x_train, y_train, x_test = load_data()

run_experiment(
    'single_xgboost',
    x_train, y_train, x_test,
    pipe, params, rmsle_score
    )

