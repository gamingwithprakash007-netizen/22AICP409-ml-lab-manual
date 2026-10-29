import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset and visualize missing values
titanic_data = pd.read_csv("../MLdatasets/titanic.csv")
sns.heatmap(titanic_data.isnull(), cbar=False)
plt.show()

sns.countplot(x='Survived', hue='Sex', data=titanic_data)
plt.show()

sns.countplot(x='Survived', hue='Pclass', data=titanic_data)
plt.show()

# Fill null values of column [Age] by mean age of people in the same Pclass
def get_age(row):
    if pd.isnull(row["Age"]):
        same_class = titanic_data[titanic_data["Pclass"] == row["Pclass"]]
        return same_class["Age"].mean()
    else:
        return row["Age"]

titanic_data["Age"] = titanic_data[["Age", "Pclass"]].apply(get_age, axis=1)
titanic_data.drop("Cabin", axis=1, inplace=True)
titanic_data.dropna(inplace=True)

sns.heatmap(titanic_data.isnull(), cbar=False)
plt.show()

# One-hot encode categorical columns
sex_data = pd.get_dummies(titanic_data["Sex"]).astype(int)
embarked_data = pd.get_dummies(titanic_data['Embarked']).astype(int)
titanic_data = pd.concat([titanic_data, sex_data, embarked_data], axis=1)
titanic_data.drop(['Name', 'PassengerId', 'Ticket', 'Sex', 'Embarked'], axis=1, inplace=True)
print(titanic_data.head(5))

# Train and evaluate Logistic Regression
X = titanic_data.drop("Survived", axis=1)
y = titanic_data["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("************Classification Report************\n")
print(classification_report(y_test, predictions))
print("***********Confusion Matrix***************")
print(confusion_matrix(y_test, predictions))
