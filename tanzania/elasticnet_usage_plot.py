from preprocessing import get_preprocessed_data
from sklearn.model_selection import GridSearchCV, train_test_split
from elasticnet_model import train_elasticnet_model
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from image import Colors, save_png

Colors.init_colors()

# Load data and split
df = get_preprocessed_data('data/Train.csv')
X = df.drop('total_cost', axis=1)
y = df['total_cost']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = train_elasticnet_model(X_train, y_train)

# Predict and calculate residuals
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5, color = Colors.accent_6)
plt.axhline(y=0, color=Colors.link, linestyle='--', linewidth=1)
plt.title('Residual Plot')
plt.xlabel('Predicted Total Cost')
plt.ylabel('Residuals')
plt.grid(True)
plt.tight_layout()
plt.show()