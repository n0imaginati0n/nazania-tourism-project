from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def knn_classification(X, y):
    # Pipeline: scaling + KNN
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    # Define hyperparameter grid
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9],
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2]  # p=1: Manhattan, p=2: Euclidean
    }

    # Grid search
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
    print("Best KNN Parameters:", best_params)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    return best_model, predictions, accuracy, best_params
