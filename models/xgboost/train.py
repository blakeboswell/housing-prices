from shared import load_data
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score


train_path = '/data/housing_prices/input/train.csv.gz'
test_path  = '/data/housing_prices/input/test.csv.gz'



def load_transformer(x_train: pd.DataFrame) -> ColumnTransformer:
    """
    """
    # get numeric columns and one-hot columns
    dummies_cols = [
        x for x in x_train.columns 
        if x_train[x].nunique() < 10 and x_train[x].dtype == "object"]

    numeric_cols = [
        x for x in x_train.columns 
        if x_train[x].dtype in ['int64', 'float64']]

    numerical_transformer = SimpleImputer(strategy='median')

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numeric_cols),
            ('cat', categorical_transformer, dummies_cols)
        ])
    

def tune_parameters(x_train, y_train, preprocessor, k_folds=2) -> tuple:
    """
    """
    params = {
        'model__min_child_weight': [1, 5, 8, 10],
        'model__gamma': [0.5, 1, 1.5, 2, 5],
        'model__subsample': [0.4, 0.6, 0.8, 1.0],
        'model__colsample_bytree': [0.4, 0.6, 0.8, 1.0],
        'model__max_depth': [3, 4, 5, 8, 10]
    }

    pipe = Pipeline(
        steps=[('preprocessor', preprocessor),
               ('model', XGBRegressor(n_estimators=600, tree_method='gpu_hist', gpu_id=0, random_state=0))]
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
    model = XGBRegressor(
        n_estimators=600, random_state=0, tree_method='gpu_hist', gpu_id=0,
        min_child_weight=min_child_weight, gamma=gamma, colsample_bytree=colsample_bytree, max_depth=max_depth
    )
    pipe = Pipeline(
        steps=[('preprocessor', preprocessor),
               ('model', model)]
    )
    kfold_scores = cross_val_score(pipe, x, y, cv=k_folds, scoring='neg_mean_absolute_error')  
    mlflow.log_metric(f"average_accuracy", -1*kfold_scores.mean())
    mlflow.log_metric(f"std_accuracy", kfold_scores.std())
      

if __name__ == '__main__':

    client = MlflowClient()
    data_name = 'test'
      
    try:
        experiment_id = client.create_experiment(data_name)
    except:
        experiment_id = client.get_experiment_by_name(data_name).experiment_id
    
    with mlflow.start_run(experiment_id=experiment_id, run_name='experiment-1-gpu'):
        
        x_train, y_train, x_test = load_data(train_path, test_path, 'SalePrice', 'Id')
        preprocessor = load_transformer(x_train)
        min_child_weight, gamma, colsample_bytree, max_depth = tune_parameters(x_train, y_train, preprocessor)
        validation(x_train, y_train, preprocessor, min_child_weight, gamma, colsample_bytree, max_depth)
