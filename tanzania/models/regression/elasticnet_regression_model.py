from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def elasticnet_regression(X, y):
    # Define hyperparameter grid
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10],
        'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0]  # 1.0 = Lasso, 0 = Ridge
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        ElasticNet(max_iter=10000),
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    # Fit the model
    grid_search.fit(X, y)

    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Predict and evaluate
    predictions = best_model.predict(X)
    rmse = mean_squared_error(y, predictions, squared=False)
    r2 = r2_score(y, predictions)

    # Output
    print("Best ElasticNet Parameters:", best_params)
    print(f"ElasticNet RMSE: {rmse:.2f}")
    print(f"ElasticNet R²: {r2:.4f}")

    return best_model, predictions, rmse, r2, best_params
