from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def gradient_boosting_regression(X, y):
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    # Fit the model
    grid_search.fit(X, y)

    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Predict using the best model
    predictions = best_model.predict(X)

    # Evaluate performance
    rmse = mean_squared_error(y, predictions, squared=False)
    r2 = r2_score(y, predictions)

    # Print results
    print("Best Gradient Boosting Parameters:", best_params)
    print(f"Gradient Boosting RMSE: {rmse:.2f}")
    print(f"Gradient Boosting R²: {r2:.4f}")

    return best_model, predictions, rmse, r2, best_params
