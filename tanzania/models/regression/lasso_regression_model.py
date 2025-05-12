from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def lasso_regression(X_train, X_test, y_train, y_test):
    # Define a range of alpha values to search over
    param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]}
    
    # Set up GridSearchCV with Lasso regression
    grid_search = GridSearchCV(Lasso(), param_grid, cv=5, scoring='neg_mean_squared_error')
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Best alpha from the grid search
    best_alpha = grid_search.best_params_['alpha']
    best_model = grid_search.best_estimator_

    # Predict with the best model
    predictions = best_model.predict(X_test)
    
    # Evaluate performance
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    
    # Print results
    print(f"Best alpha found: {best_alpha}")
    print(f"RMSE with best alpha: {rmse:.2f}")
    print(f"R² with best alpha: {r2:.4f}")
    
    return best_model, predictions, rmse, r2
