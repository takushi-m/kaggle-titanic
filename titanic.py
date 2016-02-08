# score: 0.78947
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import PolynomialFeatures as poly
from sklearn.linear_model import LogisticRegression as lr

def makeInput(data):
    x = pd.DataFrame()
    x["agena"] = data.Age.map(lambda x:1 if math.isnan(x) else 0)
    x["farena"] = data.Fare.map(lambda x:1 if math.isnan(x) else 0)

    data.Age = data.Age.fillna(29.7)
    data.Fare = data.Fare.fillna(32.2)
    data.Cabin = data.Cabin.fillna("NA")

    x["cabinna"] = data.Cabin.map(lambda x:1 if x=="NA" else 0)
    x["sibsp"] = data.SibSp
    x["parch"] = data.Parch
    x["smallfamiliy"] = (data.SibSp+data.Parch).map(lambda x:1 if x<3 else 0)
    x["bigfamiliy"] = (data.SibSp+data.Parch).map(lambda x:1 if x>=3 else 0)
    x["age"] = data.Age.map(lambda x: (x-29.7)/13.0)
    x["age-10"] = data.Age.map(lambda x:1 if x<=10 else 0)
    x["age10-15"] = data.Age.map(lambda x:1 if x>10 and x<=15 else 0)
    x["age15-20"] = data.Age.map(lambda x:1 if x>15 and x<=20 else 0)
    x["age20-25"] = data.Age.map(lambda x:1 if x>20 and x<=25 else 0)
    x["age25-30"] = data.Age.map(lambda x:1 if x>25 and x<31 else 0)
    x["age30-"] = data.Age.map(lambda x:1 if x>30 else 0)
    x["class1"] = data.Pclass.map(lambda x:1 if x==1 else 0)
    x["class2"] = data.Pclass.map(lambda x:1 if x==2 else 0)
    x["class3"] = data.Pclass.map(lambda x:1 if x==3 else 0)
    x["male"] = data.Sex.map(lambda x:1 if x=="male" else 0)
    x["female"] = data.Sex.map(lambda x:1 if x=="female" else 0)
    x["fare"] = data.Fare.map(lambda x: (x-32.2)/49.7)
    x["fare-"] = data.Fare.map(lambda x:1 if x<20 else 0)
    x["fare+"] = data.Fare.map(lambda x:1 if x>=20 else 0)

    x["mrs"] = data.Name.map(lambda x:1 if x.lower().find("mrs")>=0 else 0)
    x["mr"] = data.Name.map(lambda x:1 if x.lower().find("mr")>=0 else 0)
    x["miss"] = data.Name.map(lambda x:1 if x.lower().find("miss")>=0 else 0)
    x["master"] = data.Name.map(lambda x:1 if x.lower().find("master")>=0 else 0)

    x["embark_C"] = data.Embarked.map(lambda x:1 if x=="C" else 0)
    x["embark_Q"] = data.Embarked.map(lambda x:1 if x=="Q" else 0)
    x["embark_S"] = data.Embarked.map(lambda x:1 if x=="S" else 0)

    #return x
    p = poly(2, interaction_only=False)
    return p.fit_transform(x)

if __name__ == "__main__":
    data = pd.read_csv("./data/train.csv")

    x = makeInput(data)
    y = data.Survived

    model = lr(C=0.2)
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
