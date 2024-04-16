# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 23:37:26 2023

@author: Jimmy
"""

# In[] Pre-processing
import HappyML.preprocessor as pp

# Dataset Loading
dataset = pp.dataset("50_Startups.csv")

# Independent/Dependent Variables Decomposition
X, Y = pp.decomposition(dataset, [0, 1, 2, 3], [4])

# Apply One Hot Encoder to Column[3] & Remove Dummy Variable Trap
X = pp.onehot_encoder(X, columns=[3], remove_trap=True)
# X = pp.remove_columns(X, [3])

# Add Constants (for Standard Library only)
# import statsmodels.api as sm
# X = sm.add_constant(X)

# Split Training vs. Testing Set
X_train, X_test, Y_train, Y_test = pp.split_train_test(X, Y, train_size=0.8)

# Feature Scaling (optional)
# X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))
# Y_train, Y_test = pp.feature_scaling(fit_ary=Y_train, transform_arys=(Y_train, Y_test))

# In[] Create Linear Regressor
from HappyML.regression import SimpleRegressor

simple_reg = SimpleRegressor()
Y_pred_simple = simple_reg.fit(X_train, Y_train).predict(X_test)

# R-Squared Score
print("Goodness of Model (R-Squared Score):", simple_reg.r_score(X_test, Y_test))

# In[] Multiple Linear Regression

# # Training 降維
#features = [0, 1,2,3, 4,5]
# features = [0, 1,3]
# model = sm.OLS(exog=X_train.iloc[:, features].astype("float"), endog=Y_train.astype("float")).fit()

# # Get Model Summary Report
# print(model.summary())

# # Performance
# print("Adjusted R-Squared: ", model.rsquared_adj)

# # In[] Predict
# Y_pred = model.predict(X_test.iloc[:, features].astype("float"))
# # Y_Pred = model.predict(X_test)

# # # Parameters of Model
# print("Regression Coefficients:", model.params.to_numpy())
# print(model.params)

# In[] Multiple Linear Regression with HappyML
from HappyML.regression import MultipleRegressor

model = MultipleRegressor()
#selected_features = model.backward_elimination(X_train, Y_train, verbose=True)
model.backward_elimination(X_train, Y_train, verbose=True)
Y_pred = model.fit(X_train, Y_train).predict(X_test)

print("Adjusted R-Squared:", model.r_score())

# In[] Cross-model Performance
from HappyML.performance import rmse

rmse_linear = rmse(Y_test, Y_pred_simple)
rmse_multi = rmse(Y_test, Y_pred)

if rmse_linear < rmse_multi:
    print(f"RMSE Linear: {rmse_linear:.4f} < RMSE Multi: {rmse_multi:.4f}. Linear smaller, WIN!!")
else:
    print(f"RMSE Linear: {rmse_linear:.4f} > RMSE Multi: {rmse_multi:.4f}. Multi smaller, WIN!!")

# In[] Check for Assumption of Regression
from HappyML.criteria import AssumptionChecker

X_train, X_test = pp.feature_scaling(fit_ary=X_train, transform_arys=(X_train, X_test))
Y_train, Y_test, Y_pred = pp.feature_scaling(fit_ary=Y_train, transform_arys=(Y_train, Y_test, Y_pred))

checker = AssumptionChecker(X_train.loc[:, model.named_features], X_test.loc[:, model.named_features], Y_train, Y_test, Y_pred)
checker.y_lim = (-4, 4)
checker.heatmap = True
checker.check_all()
