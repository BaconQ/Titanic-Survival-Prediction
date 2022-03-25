import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

titanic_data = pd.read_csv("train.csv")

#Replace missing values
titanic_data = titanic_data.drop(columns="Cabin", axis=1)
titanic_data["Age"].fillna(titanic_data["Age"].mean(), inplace=True)
titanic_data["Embarked"].fillna(titanic_data["Embarked"].mode()[0], inplace=True)

#Data spliting and organizing 
titanic_data.replace({"Sex":{"male": 0, "female":1}, "Embarked":{"S": 0, "C": 1, "Q":2}}, inplace=True)
x = titanic_data.drop(columns=["PassengerId","Name", "Ticket", "Survived"], axis=1)
y = titanic_data["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=2)

#Train the Logistic Regression Model
model = LogisticRegression()
model.fit(x_train, y_train)


#Evaluate model
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)

print("Accuracy of training: ", training_data_accuracy)
print("Auccarcy of test: ", test_data_accuracy)