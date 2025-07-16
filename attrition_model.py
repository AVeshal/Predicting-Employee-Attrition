# attrition_model.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load Dataset
print("Loading data...")
df = pd.read_csv("HR-Employee-Attrition.csv")
print("Data loaded successfully!")

# Step 2: Basic Cleaning & Info
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Step 3: Visualize Attrition Distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='Attrition', palette='Set2')
plt.title("Employee Attrition Distribution")
plt.savefig("attrition_distribution.png")
plt.show()

# Step 4: Encode Categorical Variables
print("Encoding categorical variables...")
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Step 5: Feature Engineering
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 7: Build & Tune Model
print("\nTraining Random Forest model with GridSearchCV...")
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)
model = grid.best_estimator_

# Step 8: Model Evaluation
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 9: Feature Importance Plot
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(12, 6))
sns.barplot(x=importances[indices][:10], y=features[indices][:10], palette='coolwarm')
plt.title("Top 10 Important Features Influencing Attrition")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# Final Message
print("\nDone! Model is trained and evaluated.")
print("Visualizations saved: 'attrition_distribution.png', 'feature_importance.png'")
