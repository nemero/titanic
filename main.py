import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('./data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv("data/train.csv")
print(train_data.head())

test_data = pd.read_csv("data/test.csv")
print(test_data.head())

###  one train rate women who survived:
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women, sum(women), len(women))

### second train rate men who survived:
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men, sum(men), len(men))

### step Three learn net
from sklearn.ensemble import RandomForestClassifier

train_data = train_data[train_data['Embarked'].notnull()] # exclude nan rows
test_data = test_data[test_data['Embarked'].notnull()]
train_data = train_data.fillna(0) # fill nan fields to 0
test_data = test_data.fillna(0)

y = train_data["Survived"]

features = ["Sex", "Pclass", "SibSp", "Parch", "Age", "Embarked"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
#print(test_data, predictions, np.column_stack((test_data, predictions)))
#output = pd.DataFrame(np.column_stack((test_data, predictions)))
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions, 'Sex': test_data.Sex, 'Age': test_data.Age, 'Parch': test_data.Parch, 'Pclass': test_data.Pclass, 'SibSp': test_data.SibSp, 'Embarked': test_data.Embarked})
output.to_csv('data/prediction.csv', index=False)

print("Your submission was successfully saved!")