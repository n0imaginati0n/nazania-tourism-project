from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression(X,y):
    # Train the model
    model = LinearRegression().fit(X, y)
    
    # Predict on test data
    predictions = model.predict(X)
    
    # Evaluate
    rmse = mean_squared_error(y, predictions, squared=False)
    r2 = r2_score(y, predictions)
    
    # Print results
    print(f"Linear Regression RMSE: {rmse:.2f}")
    print(f"Linear Regression R²: {r2:.4f}")
    
    return model, predictions, rmse, r2
