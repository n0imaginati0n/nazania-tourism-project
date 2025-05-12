from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def decision_tree_regression(X_train, X_test, y_train, y_test):
    # Define the hyperparameter grid
    param_grid = {
        'max_depth': [None, 3, 5, 10, 20],
        'min_samples_split': [2, 5, 10, 20]
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        DecisionTreeRegressor(),
        param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='neg_mean_squared_error',  # minimize RMSE
        n_jobs=-1
    )

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Predict using the best model
    predictions = best_model.predict(X_test)

    # Evaluate performance
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)

    # Print results
    print("Best Parameters from Grid Search:", best_params)
    print(f"Best Decision Tree RMSE: {rmse:.2f}")
    print(f"Best Decision Tree R²: {r2:.4f}")

    return best_model, predictions, rmse, r2, best_params
