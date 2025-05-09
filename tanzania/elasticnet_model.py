import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


def train_elasticnet_model(X: pd.DataFrame, y: pd.Series):
    """
    Train ElasticNet model with GridSearchCV and return the best estimator.

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable (total_cost)

    Returns:
        Trained pipeline (best ElasticNet model)
    """
    # Split the data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet())
    ])

    # Define hyperparameter grid
    param_grid = {
        'model__alpha': [0.1, 1.0, 10.0],
        'model__l1_ratio': [0.2, 0.5, 0.8]
    }

    # Grid search with cross-validation
    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Evaluate model on test set
    y_pred = grid_search.predict(X_test)
    print("Best Parameters:", grid_search.best_params_)
    print("R2 Score:", round(r2_score(y_test, y_pred), 4))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))

    return grid_search.best_estimator_


def predict_total_cost(model, X_new: pd.DataFrame):
    """
    Predict total cost using the trained model.

    Args:
        model: Trained pipeline from train_elasticnet_model
        X_new (pd.DataFrame): Preprocessed feature matrix for new data

    Returns:
        np.ndarray: Predicted values
    """
    return model.predict(X_new)

