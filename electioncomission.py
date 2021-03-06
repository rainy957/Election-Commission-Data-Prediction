#minor project prediction of candidate win
#dataset :2014-2016 election commission dataset

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def notNan(num):
    return num == num

def classification_model(model, data, predictors, outcome):
    model.fit(data[predictors],data[outcome])
    predictions = model.predict(data[predictors])
    accuracy = metrics.accuracy_score(predictions,data[outcome])
    print("Accuracy of "+outcome[0]+" : %s" % "{0:.3%}".format(accuracy))
    kf = KFold(data.shape[0], n_folds=5)
    error = []
    for train, test in kf:
        train_predictors = (data[predictors].iloc[train,:])
        train_target = data[outcome].iloc[train]
        model.fit(train_predictors, train_target)
        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
        print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
    model.fit(data[predictors],data[outcome])
    print

data = pd.read_csv("elec50.csv")

data = data.drop(["Unnamed: 13"],axis=1)
data = data.drop(["Unnamed: 14"],axis=1)
data = data.drop(["Unnamed: 15"],axis=1)
data = data.drop(["Unnamed: 16"],axis=1)
data = data.drop(["Unnamed: 17"],axis=1)
data = data.drop(["Unnamed: 18"],axis=1)
data = data.drop(["Unnamed: 19"],axis=1)
data = data.drop(["Unnamed: 20"],axis=1)
data = data.drop(["Unnamed: 21"],axis=1)
data = data.drop(["Unnamed: 22"],axis=1)
data = data.drop(["Unnamed: 23"],axis=1)
data = data.drop(["Unnamed: 24"],axis=1)
data = data.drop(["Unnamed: 25"],axis=1)
data = data.drop(["Unnamed: 26"],axis=1)
data = data.drop(["Unnamed: 27"],axis=1)
data = data.drop(["Unnamed: 28"],axis=1)
data = data.drop(["Unnamed: 29"],axis=1)
data = data.drop(["Unnamed: 30"],axis=1)
data = data.drop(["Unnamed: 31"],axis=1)
data = data.drop(["Unnamed: 32"],axis=1)
data = data.drop(["Unnamed: 33"],axis=1)
data = data.drop(["Party_cd"],axis=1)
#data = data.drop(["candi_des"],axis=1)
data = data.drop(["nomi_status"],axis=1)
data = data.drop(["FiNIL_lock"],axis=1)

data = data[notNan(data.votes)]
data = data[notNan(data.Ward_no)]



print data.gender.value_counts()
data["gender"] = LabelEncoder().fit_transform(data["gender"].astype(str))
print data.gender.value_counts()

#plt.hist(data["gender"])
#plt.show()

#plt.scatter(data["age"],data["gender"])
#plt.show()

#data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
#plt.show()

#data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
#plt.show()

print data.head(49)

data["candi_des"] = LabelEncoder().fit_transform(data["candi_des"].astype(str))
data["cat"] = LabelEncoder().fit_transform(data["cat"].astype(str))
data["OccuPation"] = LabelEncoder().fit_transform(data["OccuPation"].astype(str))
data["Education"] = LabelEncoder().fit_transform(data["Education"].astype(str))
data["P.G..rr_Status"] = LabelEncoder().fit_transform(data["P.G..rr_Status"].astype(str))

print data.head(49)
#print data.describe()
#print len(data)

train,test = train_test_split(data,test_size=0.3)


predictor_var = ['age','candi_des','Ward_no','cat','votes','Education','OccuPation','P.G..rr_Status']
outcome_var = ['gender']
model = LogisticRegression()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['gender','candi_des','Ward_no','cat','votes','Education','OccuPation','P.G..rr_Status']
outcome_var = ['age']
model = LogisticRegression()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','gender','Ward_no','cat','votes','Education','OccuPation','P.G..rr_Status']
outcome_var = ['candi_des']
model = LogisticRegression()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','candi_des','gender','cat','votes','Education','OccuPation','P.G..rr_Status']
outcome_var = ['Ward_no']
model = LogisticRegression()
classification_model(model,train,predictor_var,outcome_var)


predictor_var = ['age','candi_des','Ward_no','cat','votes','Education','OccuPation','P.G..rr_Status']
outcome_var = ['cat']
model = LogisticRegression()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['gender','candi_des','Ward_no','cat','age','Education','OccuPation','P.G..rr_Status']
outcome_var = ['votes']
model = LogisticRegression()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','gender','Ward_no','cat','votes','candi_des','OccuPation','P.G..rr_Status']
outcome_var = ['Education']
model = LogisticRegression()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','candi_des','gender','cat','votes','Education','Ward_no','P.G..rr_Status']
outcome_var = ['OccuPation']
model = LogisticRegression()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','candi_des','gender','cat','votes','Education','Ward_no','OccuPation']
outcome_var = ['P.G..rr_Status']
model = LogisticRegression()
classification_model(model,train,predictor_var,outcome_var)


predictor_var = ['age','candi_des','Ward_no','cat','votes','Education','OccuPation','P.G..rr_Status']
outcome_var = ['gender']
model = GaussianNB()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['gender','candi_des','Ward_no','cat','votes','Education','OccuPation','P.G..rr_Status']
outcome_var = ['age']
model = GaussianNB()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','gender','Ward_no','cat','votes','Education','OccuPation','P.G..rr_Status']
outcome_var = ['candi_des']
model = GaussianNB()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','candi_des','gender','cat','votes','Education','OccuPation','P.G..rr_Status']
outcome_var = ['Ward_no']
model = GaussianNB()
classification_model(model,train,predictor_var,outcome_var)


predictor_var = ['age','candi_des','Ward_no','cat','votes','Education','OccuPation','P.G..rr_Status']
outcome_var = ['cat']
model = GaussianNB()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['gender','candi_des','Ward_no','cat','age','Education','OccuPation','P.G..rr_Status']
outcome_var = ['votes']
model = GaussianNB()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','gender','Ward_no','cat','votes','candi_des','OccuPation','P.G..rr_Status']
outcome_var = ['Education']
model = GaussianNB()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','candi_des','gender','cat','votes','Education','Ward_no','P.G..rr_Status']
outcome_var = ['OccuPation']
model = GaussianNB()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','candi_des','gender','cat','votes','Education','Ward_no','OccuPation']
outcome_var = ['P.G..rr_Status']
model = GaussianNB()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','candi_des','Ward_no','cat','votes','Education','OccuPation','P.G..rr_Status']
outcome_var = ['gender']
model = RandomForestClassifier()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['gender','candi_des','Ward_no','cat','votes','Education','OccuPation','P.G..rr_Status']
outcome_var = ['age']
model = RandomForestClassifier()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','gender','Ward_no','cat','votes','Education','OccuPation','P.G..rr_Status']
outcome_var = ['candi_des']
model = RandomForestClassifier()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','candi_des','gender','cat','votes','Education','OccuPation','P.G..rr_Status']
outcome_var = ['Ward_no']
model = RandomForestClassifier()
classification_model(model,train,predictor_var,outcome_var)


predictor_var = ['age','candi_des','Ward_no','cat','votes','Education','OccuPation','P.G..rr_Status']
outcome_var = ['cat']
model = RandomForestClassifier()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['gender','candi_des','Ward_no','cat','age','Education','OccuPation','P.G..rr_Status']
outcome_var = ['votes']
model = RandomForestClassifier()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','gender','Ward_no','cat','votes','candi_des','OccuPation','P.G..rr_Status']
outcome_var = ['Education']
model = RandomForestClassifier()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','candi_des','gender','cat','votes','Education','Ward_no','P.G..rr_Status']
outcome_var = ['OccuPation']
model = RandomForestClassifier()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','candi_des','gender','cat','votes','Education','Ward_no','OccuPation']
outcome_var = ['P.G..rr_Status']
model = RandomForestClassifier()
classification_model(model,train,predictor_var,outcome_var)


predictor_var = ['age','candi_des','Ward_no','cat','votes','Education','OccuPation','P.G..rr_Status']
outcome_var = ['gender']
model = DecisionTreeClassifier()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['gender','candi_des','Ward_no','cat','votes','Education','OccuPation','P.G..rr_Status']
outcome_var = ['age']
model = DecisionTreeClassifier()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','gender','Ward_no','cat','votes','Education','OccuPation','P.G..rr_Status']
outcome_var = ['candi_des']
model = DecisionTreeClassifier()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','candi_des','gender','cat','votes','Education','OccuPation','P.G..rr_Status']
outcome_var = ['Ward_no']
model = DecisionTreeClassifier()
classification_model(model,train,predictor_var,outcome_var)


predictor_var = ['age','candi_des','Ward_no','cat','votes','Education','OccuPation','P.G..rr_Status']
outcome_var = ['cat']
model = DecisionTreeClassifier()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['gender','candi_des','Ward_no','cat','age','Education','OccuPation','P.G..rr_Status']
outcome_var = ['votes']
model = DecisionTreeClassifier()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','gender','Ward_no','cat','votes','candi_des','OccuPation','P.G..rr_Status']
outcome_var = ['Education']
model = DecisionTreeClassifier()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','candi_des','gender','cat','votes','Education','Ward_no','P.G..rr_Status']
outcome_var = ['OccuPation']
model = DecisionTreeClassifier()
classification_model(model,train,predictor_var,outcome_var)

predictor_var = ['age','candi_des','gender','cat','votes','Education','Ward_no','OccuPation']
outcome_var = ['P.G..rr_Status']
model = DecisionTreeClassifier()
classification_model(model,train,predictor_var,outcome_var)

