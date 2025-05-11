#!

import warnings 
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, FunctionTransformer, RobustScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_squared_error

from tanzania import extend_train_test_with_countries, save_png, Colors

class AddSumFeature(BaseEstimator, TransformerMixin):
    def __init__(self, col_src: tuple, col_tgt: str):
        self.col_src = col_src
        self.col_tgt = col_tgt

    def fit(self, X, y = None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        X[self.col_tgt] = X[self.col_src].sum(axis = 1)
        return X

    def set_output(self, transform: str = None):
        return self

def create_model_report(name: str, y_test: pd.DataFrame, y_pred: pd.DataFrame):
    print(f'{name} model:')
    r2_test = r2_score(y_test, y_pred)
    rmse_test = mean_squared_error(y_test, y_pred) ** 0.5
    print(f'RMSE = {rmse_test.round(3)}, R2_score = {r2_test.round(5)}')

    # make a graph
    residuals = y_test - y_pred

    plt.figure(figsize = (10, 6))
    df_test = pd.DataFrame({
        'real': y_test,
        'predicted': y_pred,
        'residuals': residuals
    })

    sns.scatterplot(data = df_test, y = 'residuals', x = 'predicted', color = Colors.dark_matcha_green)
    plt.axhline(0, color = Colors.accent_4, linestyle='--', linewidth = 3)
    plt.xlabel('Predicted')
    plt.ylabel('Residual')
    plt.title('Pipeline residuals plot')

    save_png(f'pics/{name}_model_performance.png')
    plt.show()

if __name__ == '__main__':
    Colors.init_colors()

    X = extend_train_test_with_countries(pd.read_csv('data/Train.csv'))
    X.drop('ID', axis = 1, inplace = True)

    y = X.pop('total_cost')

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size= 0.75)

    X_train = X_train.join(y_train)
    # drop all the rows from dataset, wehere 'travel_with' = np.nan and total_male + total_female != 1. 20 items
    mask = X_train['travel_with'].isna() & ((X_train['total_male'] + X_train['total_female']) != 1)
    print(f'from {X_train.shape[0]} will remove {mask.sum()} because "travel_with" is unknown and number of tourists not 1')

    X_train = X_train.drop(X_train[mask].index)

    print(f'{X_train.shape[0]} rows left')

    # drop all the rows, where total_male or total_female = np.nan
    X_train = X_train.dropna(subset=['total_male', 'total_female'])
    print(f'after not defined total_male and total_female {X_train.shape[0]} rows left')

    y_train = X_train.pop('total_cost')

    num_cols = [ col for col in X_train.columns if (X_train[col].dtype == 'int') or (X_train[col].dtype == 'float') ]
    cat_cols = [ col for col in X_train.columns if X_train[col].dtype == 'object' ]
    print(f'Numerical: {len(num_cols)}, Categorical: {len(cat_cols)}, Initially: {len(X_train.columns)}')

    pipeline_most_impressing = Pipeline([
        ('value_impress', SimpleImputer(strategy='constant', fill_value = 'No comments')),
        ('onehot_impress', OneHotEncoder(handle_unknown='ignore'))
    ])

    pipeline_travel_with = Pipeline([
        ('value_travel', SimpleImputer(strategy='constant', fill_value = 'Alone')),
        ('onehot_travel', OneHotEncoder(handle_unknown='ignore'))
    ])

    pipeline_cat_other = Pipeline([
        ('imput_cats', SimpleImputer(strategy='most_frequent')),
        ('onehot_other', OneHotEncoder(handle_unknown='ignore'))
    ])

    pipeline_numeric = Pipeline([
        ('imput_num', SimpleImputer(strategy='most_frequent').set_output(transform="pandas")),
        ('add_nights', Pipeline([
            ('nights', AddSumFeature(['night_mainland', 'night_zanzibar'], 'nights'))
            ])
        ),
        ('add_people', Pipeline([
            ('people', AddSumFeature(['total_male', 'total_female'], 'people'))
            ])
        ),
        ('scale_num', StandardScaler())
    ])

    process_columns = ColumnTransformer([
        ('impressing', pipeline_most_impressing, ['most_impressing']),
        ('with', pipeline_travel_with, ['travel_with']),
        ('cats', pipeline_cat_other, [col for col in cat_cols if col not in ['most_impressing', 'travel_with']]),
        ('nums', pipeline_numeric, num_cols)
    ])

    estimators = [
        # ('Linear Regressor', LinearRegression(), {}),
        # ('Elastic Net', ElasticNet(), {
        #     'alpha': [0.01, 0.1, 1, 10],
        #     'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1]
        #     }),
        # ('KNeighborsRegressor', KNeighborsRegressor(n_jobs = -1), {
        #     'n_neighbors': [3, 5, 7, 10],
        #     'weights': ['uniform', 'distance'],
        #     'p': [1, 2, 3]
        # }),
        # ('Decision Tree Regressor', DecisionTreeRegressor(max_features = 'sqrt'), {
        #     'max_depth': [2, 5, 10, 15, 20],
        #     'min_samples_split': range(2, 5),
        #     'min_samples_leaf': [1, 2]}),
        ('Random Forest Regressor', RandomForestRegressor(n_jobs = -1, bootstrap = True, max_features = 'sqrt'), {
            'max_depth': range(10, 20, 2),
            'n_estimators': range(10, 100, 10)}),
        # ('Ada Boost Regressor', AdaBoostRegressor(), {
        #     'n_estimators': [50, 70, 90, 110, 150, 200],
        #     'learning_rate': [0.01, 0.05, 0.1, 0.5, 1],
        #     'loss': ['linear', 'square', 'exponential']}),
        ('Gradient Boosting Regressor', GradientBoostingRegressor(), {
            'n_estimators': [90, 100, 110, 120, 150, 200],
            'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
            'max_depth': range(4, 7),
            'max_features': ['sqrt', 'log2']}),
        # ('SGD Regressor', SGDRegressor(max_iter = 5000), {
        #     'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
        #     'penalty': ['l2', 'l1', 'elasticnet'],
        #     'l1_ratio': [0.1, 0.5, 0.9],
        # })
    ]

    results = []
    for name, est, params in estimators:
        y_pred_check = y_test
        
        if len(params) == 0:
            ppline = Pipeline([
                ('prep', process_columns),
                ('model', est)
            ])
            ppline.fit(X_train, y_train)
            y_pred_check = ppline.predict(X_test)

            results.append([
                name, 
                r2_score(y_test, y_pred_check).round(4),
                (mean_squared_error(y_test, y_pred_check) ** 0.5).round(2),
                {}])
        else:
            ppline = Pipeline([
                ('prep', process_columns),
                ('model', est)
            ])

            p_grid = {}
            for key, val in params.items():
                p_grid[f'model__{key}'] = val

            gs = GridSearchCV(
                estimator = ppline,
                param_grid = p_grid,
                n_jobs = -1,
                cv = 5,
                scoring = 'r2' #‘neg_root_mean_squared_error’
            )

            gs.fit(X_train, y_train)
            
            y_pred_check = gs.best_estimator_.predict(X_test)
            results.append([
                name, 
                r2_score(y_test, y_pred_check).round(4),
                (mean_squared_error(y_test, y_pred_check) ** 0.5).round(2),
                gs.best_params_])
        
        print(f'{name:35} R2: {r2_score(y_test, y_pred_check).round(4)}')

    print('===' * 30)
    for r in results:
        print(f'{r[0]:35} R2: {r[1]}, RMSE: {r[2]}, best params: {r[3]}')

