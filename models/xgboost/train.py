
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score


train_path = '/data/housing_prices/input/train.csv.gz'
test_path  = '/data/housing_prices/input/test.csv.gz'


def load_data(train_path, test_path, y_col, id_col) -> tuple:
    """
    """
    x_train = (pd.read_csv(train_path)
        .dropna(subset=y_col)
        .set_index(id_col)
    )
    y_train = x_train[y_col]
    x_train.drop([y_col], axis=1, inplace=True)
    x_test = (pd.read_csv(test_path)
        .set_index(id_col)
    )

    mlflow.log_artifact(f"{train_path}")
    mlflow.log_artifact(f"{test_path}")

    return x_train, y_train, x_test


def trans_missing(x_train: pd.DataFrame) -> ColumnTransformer:
    """
    """
    none_tran = SimpleImputer(strategy='constant',  fill_val='None')
    none_cols = [
        'alley', 'masvnrtype', 'garagetype', 'miscfeature'
        , 'bsmtqual', 'bsmtcond', 'bsmtexposure', 'bsmtfintype1', 'bsmtfintype2'
        , 'fireplacequ'
        , 'garagefinish', 'garagequal', 'garagecond'
        , 'poolqc'
        , 'fence'
    ]

    zero_tran = SimpleImputer(strategy='constant',  fill_val=0.0)
    zero_cols = ['masvnrarea', 'garageyrblt']

    mode_tran = SimpleImputer(strategy='most_frequent')
    mode_cols = ['electrical']

    median_tran = SimpleImputer(strategy='most_frequent')
    median_cols = ['lotfrontage']

    return ColumnTransformer(
        transformers=[
            ('none', none_tran, none_cols)
            ('zero', zero_tran, zero_cols),
            ('cat', mode_tran, mode_cols),
            ('median', median_tran, median_cols)
        ])


def trans_prep():



    # # get numeric columns and one-hot columns
    # dummies_cols = [
    #     x for x in x_train.columns 
    #     if x_train[x].nunique() < 10 and x_train[x].dtype == "object"]

    # numeric_cols = [
    #     x for x in x_train.columns 
    #     if x_train[x].dtype in ['int64', 'float64']]

    # numerical_transformer = SimpleImputer(strategy='median')

    # categorical_transformer = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='most_frequent')),
    #     ('onehot', OneHotEncoder(handle_unknown='ignore'))
    # ])

    # return ColumnTransformer(
    #     transformers=[
    #         ('num', numerical_transformer, numeric_cols),
    #         ('cat', categorical_transformer, dummies_cols)
    #     ])
    

def tune_parameters(x_train, y_train, preprocessor, k_folds=5) -> tuple:
    """
    """

    params = {
        'model__min_child_weight': [1], #5, 10],
        'model__gamma': [0.5], #1, 1.5, 2, 5],
        'model__subsample': [0.6], #0.8, 1.0],
        'model__colsample_bytree': [0.6], #0.8, 1.0],
        'model__max_depth': [3], #4, 5, 8]
    }

    pipe = Pipeline(
        steps=[('preprocessor', preprocessor),
               ('model', XGBClassifier(n_estimators=600))]
    )
    
    grid = RandomizedSearchCV(pipe, params, cv=k_folds)
    grid.fit(x_train, y_train)
    
    min_child_weight = grid.best_params_['model__min_child_weight']
    gamma = grid.best_params_['model__gamma']
    colsample_bytree = grid.best_params_['model__colsample_bytree']
    max_depth = grid.best_params_['model__max_depth']

    mlflow.log_param(f'best_min_child_weight', min_child_weight)
    mlflow.log_param(f'best_gamma', gamma)
    mlflow.log_param(f'best_colsample_bytree', colsample_bytree)
    mlflow.log_param(f'best_max_depth', max_depth)

    return min_child_weight, gamma, colsample_bytree, max_depth


def validation(x, y, preprocessor, min_child_weight, gamma, colsample_bytree, max_depth, k_folds=5) -> None:
    """
    """
    model = XGBClassifier(
        n_estimators=600, 
        min_child_weight=min_child_weight, gamma=gamma, colsample_bytree=colsample_bytree, max_depth=max_depth
    )
    pipe = Pipeline(
        steps=[('preprocessor', preprocessor),
               ('model', model)]
    )
    kfold_scores = cross_val_score(pipe, x, y, cv=k_folds)  
    mlflow.log_metric(f"average_accuracy", kfold_scores.mean())
    mlflow.log_metric(f"std_accuracy", kfold_scores.std())
      

if __name__ == '__main__':

    client = MlflowClient()
    data_name = 'test'
      
    try:
        experiment_id = client.create_experiment(data_name)
    except:
        experiment_id = client.get_experiment_by_name(data_name).experiment_id
    
    with mlflow.start_run(experiment_id=experiment_id, run_name='test'):
        
        x_train, y_train, x_test = load_data(train_path, test_path, 'SalePrice', 'Id')
        preprocessor = load_transformer(x_train)

        min_child_weight, gamma, colsample_bytree, max_depth = tune_parameters(x_train, y_train, preprocessor)
        validation(x_train, y_train, preprocessor, min_child_weight, gamma, colsample_bytree, max_depth)
