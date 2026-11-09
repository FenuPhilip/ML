# EXPERIMENT 3: NAIVE BAYES CLASSIFIER

# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create Naive Bayes model
model = GaussianNB()

# Step 5: Train model
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Actual Values:", y_test)
print("Predicted Values:", y_pred)
print("Accuracy:", accuracy * 100, "%")

# Step 8: Display confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 9: Display classification report
print(classification_report(
    y_test, y_pred, target_names=iris.target_names
))