from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def knn_regression(X, y):
    # Define a pipeline with scaling and KNN regressor
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor())
    ])

    # Define the hyperparameter grid
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9],
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2]  # p=1: Manhattan distance, p=2: Euclidean distance
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    # Fit the model
    grid_search.fit(X, y)

    # Best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Predict using the best model
    predictions = best_model.predict(X)

    # Evaluate performance
    rmse = mean_squared_error(y, predictions, squared=False)
    r2 = r2_score(y, predictions)

    # Print results
    print("Best KNN Parameters:", best_params)
    print(f"KNN Regression RMSE: {rmse:.2f}")
    print(f"KNN Regression R²: {r2:.4f}")

    return best_model, predictions, rmse, r2, best_params
