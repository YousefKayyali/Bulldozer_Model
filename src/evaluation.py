import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

bulldozer_train = pd.read_csv("..\\data\\bulldozer_traing_1,1.csv", low_memory=False)
bulldozer_validate = pd.read_csv("..\\data\\bulldozer_validate_1,1.csv", low_memory=False)
X_train = bulldozer_train.drop("SalePrice", axis=1)
y_train = bulldozer_train["SalePrice"]
X_validate = bulldozer_validate.drop("SalePrice", axis=1)
y_validate = bulldozer_validate["SalePrice"]

model = joblib.load('..\\model.joblib')

# Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf)
print(f"Cross-Validation MAE Scores: {-cv_scores}")
print(f"Average CV MAE: {-np.mean(cv_scores)}")

predictions = model.predict(X_validate)
mse = mean_squared_error(y_validate, predictions)
mape = mean_absolute_percentage_error(y_validate, predictions)
mae = mean_absolute_error(y_validate, predictions)
rmsle = np.sqrt(mean_squared_log_error(y_validate, predictions))

# Print evaluation results
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}")
print(f"Root Mean Squared Log Error (RMSLE): {rmsle}")
