import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures as poly
from sklearn.linear_model import LogisticRegression as lr

def makeInput(data):
    data.Age = data.Age.fillna(29.7)
    data.Fare = data.Fare.fillna(32.2)
    x = pd.DataFrame()
    x["sibsp"] = data.SibSp
    x["parch"] = data.Parch
    x["age-20"] = data.Age.map(lambda x:1 if x<21 else 0)
    x["age20-30"] = data.Age.map(lambda x:1 if x>20 and x<31 else 0)
    x["age30-"] = data.Age.map(lambda x:1 if x>30 else 0)
    x["class1"] = data.Pclass.map(lambda x:1 if x==1 else 0)
    x["class2"] = data.Pclass.map(lambda x:1 if x==2 else 0)
    x["class3"] = data.Pclass.map(lambda x:1 if x==3 else 0)
    x["male"] = data.Sex.map(lambda x:1 if x=="male" else 0)
    x["female"] = data.Sex.map(lambda x:1 if x=="female" else 0)
    x["fare+"] = data.Fare.map(lambda x:1 if x<20 else 0)
    x["fare-"] = data.Fare.map(lambda x:1 if x>=20 else 0)

    p = poly(2, interaction_only=False)
    return p.fit_transform(x)

if __name__ == "__main__":
    data = pd.read_csv("./data/train.csv")

    x = makeInput(data)
    y = data.Survived

    model = lr(C=0.1)
    model.fit(x,y)

    test_data = pd.read_csv("./data/test.csv")
    x_test = makeInput(test_data)
    predict = model.predict(x_test)
    predict = pd.Series(predict)

    y_test = pd.DataFrame({
        "PassengerId": test_data.PassengerId
        ,"Survived": predict
    })
    y_test.to_csv("./predict.csv", index=False)
