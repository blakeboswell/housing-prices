"""
    Methods and metadata developed in ../notebooks/preprocess-base.ipynb
"""

import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer


"""load feature meta data 
""" 
flt_cols = int_cols = cat_cols = replace_lvls = impute_map = levels = None
locals().update(json.load(open('/data/housing_prices/input/feature_meta.json')))


def normalize_levels(X: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """ send to lowercase and strip whitespace from categorical
        variable values
    """
    def normalize(x):
        if x.name in cat_cols:
            return x.str.lower().replace('\s+', '', regex=True)
        return x
    return X.apply(normalize)


def replace_levels(X: pd.DataFrame, replace_map: dict) -> pd.DataFrame:
    """ correct data entry issues in categorical variable values
    """
    def replace(x):
        if x.name in replace_map:
            return x.replace(replace_map[x.name])
        return x
    return X.apply(replace)


def conditional_fill(X: pd.DataFrame, impute_map: dict) -> pd.DataFrame:
    """ fill NA in set of one or more features that should be simultaneously NA 
        based on the NA of one of the features
    """
    Xcopy = X.copy()
    def fill_column(x, na_loc, fill_val):
        try:
            return np.where(na_loc & x.isna(), fill_val[x.name], x)
        except KeyError:
            return x
    
    for k, v in impute_map:
        Xcopy = Xcopy.apply(fill_column, na_loc=Xcopy[k].isna(), fill_val=v)

    return Xcopy


def ordinal_encoder(X:pd.DataFrame, encodings: dict) -> pd.DataFrame:
    """ encode variables as ordinal according to order specified in
        Kaggle provided data dictionary
    """
    Xcopy = X.copy()
    for k, levels in encodings.items():
        encoding = {lvl: i for i, lvl in enumerate(levels)}
        Xcopy[k] = Xcopy[k].map(encoding)
    return Xcopy


baseline_trans = Pipeline(steps=[
    ('txtnorm_trans', 
        Pipeline(steps=[
            ('step1', FunctionTransformer(normalize_levels, kw_args={'cat_cols': cat_cols})), 
            ('step2', FunctionTransformer(replace_levels, kw_args={'replace_map': replace_lvls}))
        ])
    ),
    ('impute_step1', 
        FunctionTransformer(conditional_fill, kw_args={'impute_map': impute_map})
    ),
    ('ordinal_trans', 
        FunctionTransformer(ordinal_encoder, kw_args={'encodings': levels})
    ),
    ('impute_stage2',
        ColumnTransformer([
            ('median', SimpleImputer(strategy='median'), ['LotFrontage']),
            ('mode', SimpleImputer(strategy='most_frequent'), ['BsmtExposure', 'BsmtFinType2', 'Electrical'])
            ], 
            remainder='passthrough'
        )
    )
])
