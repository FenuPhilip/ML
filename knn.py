# EXPERIMENT 2: K-NEAREST NEIGHBOUR CLASSIFIER

# Step 1: Import required libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load the dataset
iris = load_iris()

X = iris.data
y = iris.target

# Step 3: Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Create the KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Step 6: Train the model
knn.fit(X_train, y_train)

# Step 7: Predict the test data
y_pred = knn.predict(X_test)

# Step 8: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Actual Values:", y_test)
print("Predicted Values:", y_pred)
print("Accuracy:", accuracy * 100, "%")

# Step 9: Classification report
print(classification_report(y_test, y_pred,
                            target_names=iris.target_names))