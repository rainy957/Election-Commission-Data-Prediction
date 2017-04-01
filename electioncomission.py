#minor project prediction of candidate win
#dataset :2014-2016 election commission dataset


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def notNan(num):
    return num == num

data = pd.read_csv("elec50.csv")
print data.head(10)
print data.describe()
print len(data)

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


data = data[notNan(data.votes)]


print data.gender.value_counts()
data["gender"] = LabelEncoder().fit_transform(data["gender"].astype(str))
print data.gender.value_counts()


#plt.hist(data["gender"])
#plt.show()


#plt.scatter(data["age"],data["gender"])
#plt.show()


data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
plt.show()
