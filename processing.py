#!/usr/bin/env python3

import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor


from sklearn.preprocessing import StandardScaler, OneHotEncoder



def main():
    X = pd.read_csv('data/Train.csv')
    X = X.drop('ID', axis = 1)
    y = X.pop('total_cost')

    print(X['country'].unique())
    return

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.3)

    # imputation
    travel_with_val     = X_train['travel_with'].mode()[0]
    most_impressing_val = X_train['most_impressing'].mode()[0]
    total_male_val      = X_train['total_male'].mode()[0]
    total_female_val    = X_train['total_female'].mode()[0]

    X_train = X_train.fillna({
                            'travel_with' : travel_with_val,
                            'total_male' : total_male_val,
                            'total_female': total_female_val,
                            'most_impressing_val': most_impressing_val
                            })
    X_test = X_test.fillna({
                            'travel_with' : travel_with_val,
                            'total_male' : total_male_val,
                            'total_female': total_female_val,
                            'most_impressing_val': most_impressing_val
                            })

    # feature engineering
    # person
    X_train['person'] = X_train['total_male'] + X_train['total_female']
    X_test['person'] = X_test['total_male'] + X_test['total_female']    

    # nights
    X_train['nights'] = X_train['night_mainland'] + X_train['night_zanzibar']
    X_test['nights'] = X_test['night_mainland'] + X_test['night_zanzibar']

    # nights
    pack_cols = [col for col in X_train.columns if 'package_' in col]
    X_train['package'] = X_train[pack_cols].apply(lambda row: row.map({'Yes' : 1, 'No' : 0}).sum(), axis = 1)
    X_test['package']  = X_test[pack_cols].apply(lambda row: row.map({'Yes' : 1, 'No' : 0}).sum(), axis = 1)



    #===============================================================================
    #   start preprocess data
    #

    num_cols = [col for col in X_train.columns if X_train[col].dtype != 'object']
    cat_cols = [col for col in X_train.columns if col not in num_cols]

    print(num_cols)
    # standard scaler
    scaler = StandardScaler().set_output(transform='pandas')
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test [num_cols] = scaler.transform(X_test[num_cols])

    # onehot encoder ignoring diffeernt counties in train-test sets
    ohe = OneHotEncoder(
        drop = 'first',
        handle_unknown = 'ignore',
        sparse_output = False)

    ohe_array = ohe.fit_transform(X_train[cat_cols])
    ohe_df = pd.DataFrame(
        ohe_array,
        columns = ohe.get_feature_names_out(cat_cols),
        index = X_train.index
    )
    X_train = X_train.drop(cat_cols, axis = 1)
    X_train = pd.concat([X_train, ohe_df], axis = 1)

    ohe_array1 = ohe.transform(X_test[cat_cols])
    ohe_df1 = pd.DataFrame(
        ohe_array1,
        columns = ohe.get_feature_names_out(cat_cols),
        index = X_test.index
    )
    X_test = X_test.drop(cat_cols, axis = 1)
    X_test = pd.concat([X_test, ohe_df1], axis = 1)
    
    #
    #   end preprocess data
    #===============================================================================


    estimators = [
        ('LinearRegression', LinearRegression(), {}),
        ('ElasticNet', ElasticNet(), {}),
        ('SGDRegressor', SGDRegressor(), {}),
        ('DecisionTreeRegressor', DecisionTreeRegressor(), {}),
        ('KNeighborsRegressor', KNeighborsRegressor(n_jobs = -1), {}),
        ('RandomForestRegressor', RandomForestRegressor(n_jobs = -1), {}),
        ('GradientBoostingRegressor', GradientBoostingRegressor(), {}),
        ('AdaBoostRegressor', AdaBoostRegressor(), {})
    ]

    for name, est, _ in estimators:
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)

        rmse = round( mean_squared_error(y_test, y_pred) ** 0.5, 2)
        r2 = round( r2_score(y_test, y_pred), 4)
        print(f'{name:30}: RMSE = {rmse:12}, R2 = {r2}')
    
    estimators_grid = [
        ('SGDRegressor', SGDRegressor(max_iter = 5000), {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1],
            'penalty': ['l2', 'l1']
        }),
        ('KNeighborsRegressor', KNeighborsRegressor(n_jobs = -1), {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance'],
            'p': [1, 2, 3]
        }),
        ('RandomForestRegressor', RandomForestRegressor(n_jobs = -1, max_features = 'sqrt'), {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [None, 10, 20, 30],
            'bootstrap': [True, False]
        }),
        ('GradientBoostingRegressor', GradientBoostingRegressor(max_features = 'sqrt'), {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [None, 10, 20, 30],
        }),
    ]

    print("\nStart Searching Better params")
    
    for name, est, parm in estimators_grid:
        gs = GridSearchCV(
            estimator = est,
            n_jobs = -1,
            cv = 5,
            scoring = 'r2',
            param_grid = parm
        )

        gs.fit(X_train, y_train)
        y_pred = gs.best_estimator_.predict(X_test)

        rmse = round( mean_squared_error(y_test, y_pred) ** 0.5, 2)
        r2 = round( r2_score(y_test, y_pred), 4)
        print(f'{name:30}: RMSE = {rmse:12}, R2 = {r2}')



if __name__ == '__main__':
    main()


