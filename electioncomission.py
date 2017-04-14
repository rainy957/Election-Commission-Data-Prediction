import pandas as pd
import numpy as np
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def notNan(num):
    return num == num

def notInRange(num):
    return num <= 100



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
data = data.drop(["candi_des"],axis=1)
data = data.drop(["nomi_status"],axis=1)
data = data.drop(["FiNIL_lock"],axis=1)
data = data.drop(["cand_NILme"],axis=1)
data = data.drop(["P.G..rr_Status"],axis=1)

data = data[notNan(data.votes)]
data = data[notNan(data.Ward_no)]
data = data[notInRange(data.age)]



"""data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

data.hist()
plt.show()

scatter_matrix(data)
plt.show()"""


data["cat"] = LabelEncoder().fit_transform(data["cat"].astype(str))
data["gender"] = LabelEncoder().fit_transform(data["gender"].astype(str))
data["OccuPation"] = LabelEncoder().fit_transform(data["OccuPation"].astype(str))
data["Education"] = LabelEncoder().fit_transform(data["Education"].astype(str))
#data["P.G..rr_Status"] = LabelEncoder().fit_transform(data["P.G..rr_Status"].astype(str))

print(data.shape)
print(data.head(20))
print(data.describe())

print(data.groupby('age').size())
print(data.groupby('gender').size())



array = data.values
X = array[:,1:]
Y = array[:,0]
validation_size = 0.40
seed = 7
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)



models = []
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []
for name, model in models:
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=5, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
	
modelsName = ['LDA','KNN','DTC','NB','SVM']

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(modelsName)
plt.show()


svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
