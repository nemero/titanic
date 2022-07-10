import numpy as np
import pandas as pd

prediction_data = pd.read_csv("data/prediction.csv")
#print(prediction_data.head())

women = prediction_data.loc[prediction_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women, sum(women), len(women))
men = prediction_data.loc[prediction_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men, sum(men), len(men))
print('')

for am in range(1, 4):
    pclass = prediction_data.loc[prediction_data.Pclass == am]["Survived"]
    rate_pclass = sum(pclass)/len(pclass) if len(pclass) > 0 else 0.0
    print("% of pclass {0} who survived: {1}, {2}/{3}".format(am, rate_pclass, sum(pclass), len(pclass))) if len(pclass) > 0 else 0
print('')

for am in range(1, 7):
    pclass = prediction_data.loc[prediction_data.Parch == am]["Survived"]
    rate_pclass = sum(pclass)/len(pclass) if len(pclass) > 0 else 0
    print("% of parch {0} who survived: {1}, {2}/{3}".format(am, rate_pclass, sum(pclass), len(pclass))) if len(pclass) > 0 else 0
print('')

for am in range(1, 90):
    pclass = prediction_data.loc[prediction_data.Age == am]["Survived"]
    rate_pclass = sum(pclass)/len(pclass) if len(pclass) > 0 else 0.0
    print("% of age {0} who survived: {1}, {2}/{3}".format(am, rate_pclass, sum(pclass), len(pclass))) if len(pclass) > 0 else 0
print('')

step = 10
for am in range(0, round(100/step)):
    age = am * step
    to_age = am * step + step
    pclass = prediction_data.loc[(prediction_data.Age > age) & (prediction_data.Age <= to_age)]["Survived"]
    rate_pclass = sum(pclass)/len(pclass) if len(pclass) > 0 else 0.0
    print("% of age from {0} to {4} who survived: {1}, {2}/{3}".format(age, rate_pclass, sum(pclass), len(pclass), to_age)) if len(pclass) > 0 else 0
print('')

for am in range(1, 9):
    pclass = prediction_data.loc[prediction_data.SibSp == am]["Survived"]
    rate_pclass = sum(pclass)/len(pclass) if len(pclass) > 0 else 0.0
    print("% of sibsp {0} who survived: {1}, {2}/{3}".format(am, rate_pclass, sum(pclass), len(pclass))) if len(pclass) > 0 else 0
print('')

for am in ['C', 'Q', 'S']:
    pclass = prediction_data.loc[prediction_data.Embarked == am]["Survived"]
    rate_pclass = sum(pclass)/len(pclass) if len(pclass) > 0 else 0.0
    print("% of Embarked {0} who survived: {1}, {2}/{3}".format(am, rate_pclass, sum(pclass), len(pclass)))
print('')