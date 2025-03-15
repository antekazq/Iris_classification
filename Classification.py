from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize and train classifier
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

#Predict on the test set
y_pred = model.predict(X_test)

#Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


#Making a single prediction with random data to test
predicted_class = model.predict([[0,1,3,5]])
predicted_probability = model.predict_proba([[0,1,3,5]])

#Print the predicted class
print("Predicted Class:", predicted_class)
print("Predicted Probability:", predicted_probability)