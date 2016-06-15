# ----------------
# IMPORT PACKAGES
# ----------------

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ----------------
# OBTAIN DATA
# ----------------

train = pd.read_csv("train.csv")

# ----------------
# PROFILE DATA
# ----------------

percent = 0.7

train_target = train.iloc[:int(percent * len(train)), 0]
train_data = train.iloc[:int(percent * len(train)), 1:]


validation_target = train.iloc[int(percent * len(train)):, 0]
validation_data = train.iloc[int(percent * len(train)):, 1:]

n = 100
rfc = RandomForestClassifier(n_estimators=int(n), oob_score=True)
rfc.fit(train_data, train_target)

for n in range(100, 500, 100):
	rfc = RandomForestClassifier(n_estimators=int(n), oob_score=True)
	rfc.fit(train_data, train_target)
	print("Out-Of-Bag (OOB) Score: %f" % rfc.oob_score_)
	valid_pred = rfc.predict(validation_data)
	print("Mean Accuracy score for validation data set = %f" %(rfc.score(validation_data, validation_target)))