from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def svr_regression(X, y):
    # It's important to scale features for SVR
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR())
    ])

    # Define hyperparameter grid for SVR
    param_grid = {
        'svr__C': [0.1, 1, 10],
        'svr__epsilon': [0.01, 0.1, 0.5],
        'svr__kernel': ['rbf', 'linear']
    }

    # Grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    # Fit the grid search
    grid_search.fit(X, y)

    # Best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Make predictions
    predictions = best_model.predict(X)

    # Evaluate model
    rmse = mean_squared_error(y, predictions, squared=False)
    r2 = r2_score(y, predictions)

    # Print results
    print("Best SVR Parameters:", best_params)
    print(f"SVR RMSE: {rmse:.2f}")
    print(f"SVR R²: {r2:.4f}")

    return best_model, predictions, rmse, r2, best_params
