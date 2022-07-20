import sys
sys.path.append('..')

from preprocess.load import load_data
from experiment import run_experiment
from preprocess.baseline import baseline_trans
from metrics import rmsle

import numpy as np

from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import StackingRegressor

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict



cat_selector = make_column_selector(dtype_include=object)
num_selector = make_column_selector(dtype_include=np.number)

num_linear_processor = Pipeline(steps=(
    ('standardscalar', StandardScaler()),
    ('simple_mean', SimpleImputer(strategy='mean', add_indicator=True))
))
linear_preprocessor = make_column_transformer(
    (num_linear_processor, num_selector),
    (OneHotEncoder(handle_unknown='ignore'), cat_selector)
)

lasso_pipeline = Pipeline(steps=[
    ('process', linear_preprocessor),
    ('model', LassoCV())
])

tree_preprocessor = Pipeline(steps=[
    ('process', baseline_trans), 
    ('impute', SimpleImputer(strategy='most_frequent', add_indicator=True))
])

rf_pipeline = Pipeline(steps=[
    ('process', tree_preprocessor),
    ('model', RandomForestRegressor())
])

gbdt_pipeline = Pipeline(steps=[
    ('process', tree_preprocessor),
    ('model', HistGradientBoostingRegressor())
])
    
estimators = [
    ('random_forest', rf_pipeline),
    ('lasso', lasso_pipeline),
    ('hist_boosting', gbdt_pipeline),
]

stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=RidgeCV())

rmsle_score = make_scorer(rmsle, greater_is_better=False)

params = {
    'random_forest__model__n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
    'random_forest__model__max_depth': [int(x) for x in np.linspace(10, 110, num = 11)] + [None],
    'random_forest__model__min_samples_split': [2, 5, 10],
    'random_forest__model__min_samples_leaf': [1, 2, 4],
    'random_forest__model__bootstrap': [True, False]
}

x_train, y_train, x_test = load_data()

run_experiment('regstack', x_train, y_train, x_test, stacking_regressor, params, rmsle_score)
