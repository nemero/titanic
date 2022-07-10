import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_data = pd.read_csv("data/train.csv")
print(train_data.head())

###  one train rate women who survived:
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women, sum(women), len(women))

### second train rate men who survived:
men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men, sum(men), len(men))
print('')

for am in range(1, 4):
    pclass = train_data.loc[train_data.Pclass == am]["Survived"]
    rate_pclass = sum(pclass)/len(pclass) if len(pclass) > 0 else 0.0
    print("% of pclass {0} who survived: {1}, {2}/{3}".format(am, rate_pclass, sum(pclass), len(pclass))) if len(pclass) > 0 else 0
print('')

for am in range(1, 7):
    pclass = train_data.loc[train_data.Parch == am]["Survived"]
    rate_pclass = sum(pclass)/len(pclass) if len(pclass) > 0 else 0
    print("% of parch {0} who survived: {1}, {2}/{3}".format(am, rate_pclass, sum(pclass), len(pclass))) if len(pclass) > 0 else 0
print('')

for am in range(1, 9):
    pclass = train_data.loc[train_data.SibSp == am]["Survived"]
    rate_pclass = sum(pclass)/len(pclass) if len(pclass) > 0 else 0.0
    print("% of sibsp {0} who survived: {1}, {2}/{3}".format(am, rate_pclass, sum(pclass), len(pclass))) if len(pclass) > 0 else 0
print('')

for am in range(1, 90):
    pclass = train_data.loc[train_data.Age == am]["Survived"]
    rate_pclass = sum(pclass)/len(pclass) if len(pclass) > 0 else 0.0
    print("% of age {0} who survived: {1}, {2}/{3}".format(am, rate_pclass, sum(pclass), len(pclass))) if len(pclass) > 0 else 0
print('')

step = 10
for am in range(0, round(100/step)):
    age = am * step
    to_age = am * step + step
    pclass = train_data.loc[(train_data.Age > age) & (train_data.Age <= to_age)]["Survived"]
    rate_pclass = sum(pclass)/len(pclass) if len(pclass) > 0 else 0.0
    print("% of age from {0} to {4} who survived: {1}, {2}/{3}".format(age, rate_pclass, sum(pclass), len(pclass), to_age)) if len(pclass) > 0 else 0
print('')

for am in ['C', 'Q', 'S']:
    pclass = train_data.loc[train_data.Embarked == am]["Survived"]
    rate_pclass = sum(pclass)/len(pclass) if len(pclass) > 0 else 0.0
    print("% of Embarked {0} who survived: {1}, {2}/{3}".format(am, rate_pclass, sum(pclass), len(pclass))) if len(pclass) > 0 else 0
print('')