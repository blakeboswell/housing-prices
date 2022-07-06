import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SimpleImputer


from sklearn.linear_model import Ridge, HuberRegressor, LinearRegression

from shared import ConstantMapImputer


def load_transformer(x_train: pd.DataFrame) -> ColumnTransformer:
    """
    """
    mode_impute = SimpleImputer(strategy='most_frequent')
    mode_cols = [
        'exterior1st'
        , 'exterior2nd'
        , 'masvnrtype'
        , 'electrical'
        , 'functional'
        , 'kitchenqual'
        , 'saletype'
    ]

    median_impute = SimpleImputer(strategy='median')
    median_cols = ['garageyrblt']

    zone_impute = ConstantMapImputer(map_col='neighborhood', key_vals={'IDOTRR': 'RM', 'Mitchel': 'RL'})
    zone_cols = ['mszoning']

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer(
        transformers=[
            ('mode_impute', mode_impute, mode_cols),
            ('median_impute', median_impute, median_cols)
            ('zone_impute', zone_impute, zone_cols)
        ])




train_path = '/data/housing_prices/input/train.csv.gz'
test_path  = '/data/housing_prices/input/test.csv.gz'

drop_cols = ['utilities']



median_impute = SimpleImputer(strategy='median')
median_cols = []

zero_impute = SimpleImputer(strategy='constant', fill_value=0)
zero_cols = []

none_impute = SimpleImputer(strategy='constant', fill_value="None")
none_cols = []








x_train, y_train, x_test = load_data(train_path, test_path, 'SalePrice', 'Id')


# fill zoning according to neighborhood


# x_train.loc[x_train['mszoning'].isnull() & (x_train['neighborhood'] == 'IDOTRR'), ] = 'RM'
# x_train.loc[x_train['mszoning'].isnull() & (x_train['neighborhood'] == 'Mitchel'), ] = 'RL'


# lotfrontage, fill based on linear relationship with lotarea
linreg = LinearRegression()
linreg.fit(x_train['lotarea'].values.reshape(-1, 1), x_train['lotfrontage'].values)
x_train['lotfrontage'].fillna(linreg.intercept_ + x_train['lotarea'] * linreg.coef_[0] , inplace=True)


# alley NA means No Access
none_cols += ['alley']


# Impute exterior coverings with most popular option (vinyl)


# Masonry Veneer type and area should always be simultaneously NA/Non-NA.
# that's not the case though... assume type is mode when area is present by type is NA
x_train.loc[(x_train['masvnrtype'].isna()) & (x_train['nasvnrarea'].notna()), 'masvnrtype'] = 'BrkFace'


# We assume missing basement exposure of unfinished basement is "No".
x_train.loc[(x_train['bsmtexposure'].isna() & x_train['bsmtfintype1'].notna()), 'bsmtexposure'] = 'No'

# We impute missing basement condition with "mean" value of Typical.
x_train.loc[(x_train['bsmtcond'].isna() & x_train['bsmtfintype1'].notna()), 'bsmtcond'] = 'TA'

# if basement finish type is NA then all measurements should be 0.
b1_measures = ['bsmtfintype1', 'bsmtfinsf1']
b2_measures = ['bsmtfintype2', 'bsmtfinsf2']
b_measures  = ['bsmtunfsf', 'totalbsmtsf', 'bsmtfullbath', 'bsmthalfbath']

x_train.loc[x_train['bsmtfintype1'].isna() & x_train['bsmtfintype2'].isna(), b_measures] = 0.0
x_train.loc[x_train['bsmtfintype1'].isna(), b1_measures] = 0.0
x_train.loc[x_train['bsmtfintype2'].isna(), b2_measures] = 0.0

# fill electrical, functional, and kitchenqual with most frequent


# fill missing garage car and area with 0
x_train['garagecars'].fillna(0, inplace=True)
x_train['garagearea'].fillna(0, inplace=True)



# final ...
x_train.fillna("None", inplace=True)