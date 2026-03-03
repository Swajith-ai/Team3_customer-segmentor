import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
 
# For Classification 

from sklearn.linear_model import LogisticRegression

# Load your specific dataset
# .csv file must be in the same directory as this script or provide the correct path
df= pd.read_csv("titanic.csv")
print("---RAW DATA---")
print(df.head()) # show firsst 5 rows
print(df.columns) # show column names

# check for empty spots
print(df.isnull().sum())

# strategy 1: delete rows with missing info (easiest), use when you have lots of data
df = df.dropna()

# strategy 2: fill with average (better for numbers), use when data is precious
df['PassengerId'] = df['PassengerId'].fillna(df['PassengerId'].mean())

le = LabelEncoder()

# example: converting 'gender' column
# if you have text columns, you must do this for each one
#df['PassengerId'] = le.fit_transform(df['PassengerId'])

# check if it worked
print("---CLEAN DATA---")
print(df.head()) 

# X = the inputs (everything except the target) 
# Y = the target(the thing you want to predict)

X = df.drop('Survived', axis=1)
y = df['Survived']


print(X.shape)    # should be (rows,columns)
print(y.shape)    # should be (rows,)

# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Data Size:", X_train.shape)
print("Testing Data Size:", X_test.shape)

# choose your model 
model = LogisticRegression()    

model.fit(X_train, y_train) # feeds all your training data into the model. it finds patterns, learns relationships, and builds an internal 'brain'. everything before this was just prep

print("Model trained successfully!")

# aks model to predict the 20% it hasn't seen
predictions = model.predict(X_test)

# compare with reality
from sklearn.metrics import accuracy_score, mean_squared_error

print("Accuracy:", accuracy_score(y_test, predictions))
