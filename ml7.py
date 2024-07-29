from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets
iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target
print(iris_data)
print(iris_labels)
x_train, x_test, y_train, y_test = train_test_split(iris_data, iris_labels, test_size=0.30)
clsify = KNeighborsClassifier(n_neighbors=5)
clsify.fit(x_train, y_train)
y_pred = clsify.predict(x_test)
print("Confusion matrix : \n", confusion_matrix(y_test, y_pred))
print("Classification report : \n", classification_report(y_test, y_pred)