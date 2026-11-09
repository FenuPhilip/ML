# EXPERIMENT 4: DECISION TREE USING INFORMATION GAIN

# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step 2: Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Create Decision Tree model
model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

# Step 5: Train model
model.fit(X_train, y_train)

# Step 6: Predict
y_pred = model.predict(X_test)

# Step 7: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Actual Values:", y_test)
print("Predicted Values:", y_pred)
print("Accuracy:", accuracy * 100, "%")

# Step 8: Visualize decision tree
plt.figure(figsize=(15, 10))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)

plt.title("Decision Tree using Entropy")
plt.show()