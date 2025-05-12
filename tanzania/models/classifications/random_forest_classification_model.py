from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def random_forest_classification(X, y):
    # Define the model
    model = RandomForestClassifier(random_state=42)

    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Set up Grid Search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    # Fit model
    grid_search.fit(X, y)

    # Best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Predictions and evaluation
    predictions = best_model.predict(X)
    accuracy = accuracy_score(y, predictions)
    report = classification_report(y, predictions)

    # Output
    print("Best Random Forest Parameters:", best_params)
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    return best_model, predictions, accuracy, best_params
