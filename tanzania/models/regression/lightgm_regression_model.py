from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def lightgm_regression(X_train, X_test, y_train, y_test):
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 50],
        'max_depth': [-1, 5, 10]
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        LGBMRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Make predictions
    predictions = best_model.predict(X_test)

    # Evaluate the model
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)

    # Print results
    print("Best LightGBM Parameters:", best_params)
    print(f"LightGBM RMSE: {rmse:.2f}")
    print(f"LightGBM R²: {r2:.4f}")

    return best_model, predictions, rmse, r2, best_params
