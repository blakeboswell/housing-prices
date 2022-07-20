import sys

sys.path.append('..')

import numpy as np

from experiment import run_experiment
from preprocess.load import load_data
from metrics import rmsle

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
 
from sklearn.preprocessing import OneHotEncoder

x_train, y_train, x_test = load_data()

dummies_cols = [x for x in x_train.columns 
                if x_train[x].nunique() < 10 and x_train[x].dtype == "object"]

numeric_cols = [x for x in x_train.columns 
                if x_train[x].dtype in ['int64', 'float64']]

numerical_transformer = SimpleImputer(strategy='median')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_cols),
        ('cat', categorical_transformer, dummies_cols)
    ])

params = {
    'model__n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
    'model__max_depth': [int(x) for x in np.linspace(10, 110, num = 11)] + [None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__bootstrap': [True, False]
}

pipe = Pipeline(
    steps=[('preprocessor', preprocessor),
           ('model', RandomForestRegressor())]
)

rmsle_score = make_scorer(rmsle, greater_is_better=False)

run_experiment(
    'baseline',
    x_train, y_train, x_test,
    pipe, params, rmsle_score
    )


