from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def random_forest_regression(X_train, X_test, y_train, y_test):
    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Extract best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Make predictions
    predictions = best_model.predict(X_test)

    # Evaluate performance
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)

    # Print results
    print("Best Parameters from Grid Search:", best_params)
    print(f"Random Forest RMSE: {rmse:.2f}")
    print(f"Random Forest R²: {r2:.4f}")

    return best_model, predictions, rmse, r2, best_params
