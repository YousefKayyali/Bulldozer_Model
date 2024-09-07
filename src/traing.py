from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import joblib
import seaborn as sns
from matplotlib import pyplot as plt

bulldozer_traing=pd.read_csv("..\\data\\bulldozer_traing_1,1.csv",low_memory=False)
bulldozer_validate=pd.read_csv("..\\data\\bulldozer_validate_1,1.csv",low_memory=False)
rfr=RandomForestRegressor(n_jobs=10)
rfr.fit(bulldozer_traing.drop("SalePrice",axis=1),bulldozer_traing.SalePrice)
print(rfr.score(bulldozer_validate.drop("SalePrice",axis=1),bulldozer_validate.SalePrice))
dtr=DecisionTreeRegressor()
dtr.fit(bulldozer_traing.drop("SalePrice",axis=1),bulldozer_traing.SalePrice)
print(dtr.score(bulldozer_validate.drop("SalePrice",axis=1),bulldozer_validate.SalePrice))
# I use Random search to find best hyperparameter in  notebook and this is the model build by best hyperparmeter
rfr=RandomForestRegressor(n_estimators=500,min_samples_split=5,max_features="sqrt",oob_score=False,bootstrap=True,n_jobs=11)
rfr.fit(bulldozer_traing.drop("SalePrice",axis=1),bulldozer_traing.SalePrice)
print(rfr.score(bulldozer_validate.drop("SalePrice",axis=1),bulldozer_validate.SalePrice))
fig,sub=plt.subplots()
sub.barh(y=bulldozer_traing.drop("SalePrice",axis=1).columns,width=rfr.feature_importances_*100)
fig.set_figheight(12)
fig.set_figwidth(12)
sub.set_ylabel("feature importances")
sub.set_xlabel("feature")
#joblib.dump(rfr, '..\\model.joblib')
