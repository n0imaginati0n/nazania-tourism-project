from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def decision_tree_classification(X, y):
    # Define the model
    model = DecisionTreeClassifier(random_state=42)

    # Define hyperparameter grid
    param_grid = {
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']  # or 'log_loss' for newer versions
    }

    # Grid search
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    # Fit model
    grid_search.fit(X, y)

    # Get best model and params
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Predict and evaluate
    predictions = best_model.predict(X)
    accuracy = accuracy_score(y, predictions)
    report = classification_report(y, predictions)

    # Output
    print("Best Decision Tree Parameters:", best_params)
    print(f"Decision Tree Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    return best_model, predictions, accuracy, best_params
