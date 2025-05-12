from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

def svm_classification(X, y):
    # Pipeline for scaling + SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC())
    ])

    # Define hyperparameter grid
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__kernel': ['linear', 'rbf'],
        'svm__gamma': ['scale', 'auto']  # Only relevant for RBF kernel
    }

    # Grid Search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    # Fit model
    grid_search.fit(X, y)

    # Best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Predict and evaluate
    predictions = best_model.predict(X)
    accuracy = accuracy_score(y, predictions)
    report = classification_report(y, predictions)

    # Output
    print("Best SVM Parameters:", best_params)
    print(f"SVM Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    return best_model, predictions, accuracy, best_params
