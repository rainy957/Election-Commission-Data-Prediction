#minor project prediction of candidate win
#dataset :2014-2016 election commission dataset


import pandas as pd
import numpy as np

data = pd.read_csv("elec50.csv")
print data.head(10)
print data.describe()