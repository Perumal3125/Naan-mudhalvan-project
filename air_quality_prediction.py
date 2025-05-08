import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load dataset
df = pd.read_csv('air_quality.csv')

# Step 2: Check the dataset
print(df.head())

# Step 3: Drop missing values and unnecessary columns
df = df.dropna()
df = df.select_dtypes(include=[np.number])  # Keep numeric columns

# Step 4: Define features and target (AQI or any pollutant like PM2.5)
target = 'PM2.5' if 'PM2.5' in df.columns else df.columns[-1]
X = df.drop(columns=[target])
y = df[target]

# Step 5: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Step 9: Feature importance
plt.figure(figsize=(10,5))
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.title("Feature Importance")
plt.show()
