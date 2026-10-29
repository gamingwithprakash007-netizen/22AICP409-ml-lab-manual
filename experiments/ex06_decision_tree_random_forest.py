import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load and Inspect Kyphosis Data
kyphosis_df = pd.read_csv('../MLdatasets/kyphosis.csv')
print(kyphosis_df.head())
kyphosis_df.info()

sns.pairplot(kyphosis_df, hue='Kyphosis')
plt.show()

# 2. Split Data
X = kyphosis_df.drop('Kyphosis', axis=1)
y = kyphosis_df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 3. Decision Tree Classifier
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)
decision_tree_y_pred = decision_tree_model.predict(X_test)

print("Classification Report for Decision Tree Model using Kyphosis Data:")
print(classification_report(y_test, decision_tree_y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, decision_tree_y_pred))

# 4. Random Forest Classifier
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)
random_forest_y_pred = random_forest_model.predict(X_test)

print("Classification Report for Random Forest Model using Kyphosis Data:")
print(classification_report(y_test, random_forest_y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, random_forest_y_pred))
