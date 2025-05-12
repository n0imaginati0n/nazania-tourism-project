from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def logistic_regression(X, y):
    # Create a pipeline with scaling and logistic regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(solver='liblinear', max_iter=1000))
    ])

    # Define the hyperparameter grid
    param_grid = {
        'logreg__C': [0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
        'logreg__penalty': ['l1', 'l2']        # Regularization types
    }

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
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
    accuracy = accuracy_score(y, predictions)
    report = classification_report(y, predictions)

    # Print results
    print("Best Logistic Regression Parameters:", best_params)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    return best_model, predictions, accuracy, best_params
