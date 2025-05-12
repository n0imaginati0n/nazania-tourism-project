from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression(X_train, X_test, y_train, y_test):
    # Train the model
    model = LinearRegression().fit(X_train, y_train)
    
    # Predict on test data
    predictions = model.predict(X_test)
    
    # Evaluate
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    
    # Print results
    print(f"Linear Regression RMSE: {rmse:.2f}")
    print(f"Linear Regression R²: {r2:.4f}")
    
    return model, predictions, rmse, r2
