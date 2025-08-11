import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets # datasets module contains sample datasets 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load the dataset
iris = datasets.load_iris()
X = iris.data # Input variables in feature matrix
y = iris.target # Target variable (that we want to predict)

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)

classifier = DecisionTreeClassifier(criterion="entropy",random_state=42)
classifier.fit(X_train, y_train)

plt.figure(figsize=(10,8))
plot_tree(
    classifier,# Trained model
    filled=True,
    feature_names=iris.feature_names,
    class_names=iris.target_names
)

plt.show()

accuracy = classifier.score(X_test, y_test)
print(f"Test Accuracy: {accuracy: .2f}")

# Interpreting the decision boundaries
X = iris.data[:,[2,3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
classifier = DecisionTreeClassifier(criterion="entropy", random_state=42)
classifier.fit(X_train, y_train)

x_min, x_max = X[:, 0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10,8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.rainbow)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=50, cmap=plt.cm.rainbow)
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.title("Decision Tree Decision Boundaries")
plt.show()
