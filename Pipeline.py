import pandas as pd
import statsmodels.formula.api as sm
from math import log
from numpy import sort
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
# Models
from sklearn.linear_model import RandomizedLasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV


def modelSelector(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.33,
                                                        random_state=7)
    model = LassoCV()
    model.fit(X_train, y_train)
    # make predictions for test data and evaluate
    y_pred = model.predict(X_test)
    RMSE = mean_squared_error(map(log, y_test), map(log, y_pred))**0.5
    # Fit model using each importance as a threshold
    selectedModel = SelectFromModel(model, threshold=0.0, prefit=True)
    minRMSE = RMSE
    thresholds = sort(model.coef_)
    for thresh in thresholds:
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(X_train)
        # train model
        selection_model = LassoCV()
        selection_model.fit(select_X_train, y_train)
        # eval model
        select_X_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_X_test)
        try:
            RMSE = mean_squared_error(y_test, y_pred)**0.5
            if RMSE <= minRMSE:
                selectedModel = selection
                minRMSE = RMSE
        except ValueError:
            print("Invalid number for log")
    return selectedModel


def modelCreator(X, Y):
    # These are the models used for initial modelling
    models = [LassoCV(), Lasso(), RandomForestRegressor()]
    fittedModels = []
    for m in models:
        fittedModels.append(m.fit(X, Y))
    return fittedModels


def combineModels(fittedModels, X, y):
    df_y = pd.DataFrame()
    for i, m in enumerate(fittedModels):
        print fittedModels
        df_y['model'+str(i)] = m.predict(X)
    finalModel = RandomForestRegressor()
    finalModel.fit(df_y, y)
    return finalModel


def applyModels(models, X):
    df_y = pd.DataFrame()
    for i, m in enumerate(models):
        df_y['model'+str(i)] = m.predict(X)
    return df_y.mean(axis=1)


# The categorical values have been converted to numerical dummy variables
df_Dummy = pd.get_dummies(pd.read_csv('train.csv')).dropna().drop('Id', 1)
df_test = pd.get_dummies(pd.read_csv('test.csv'))
df_test_id = df_test.Id.copy()
df_test = df_test.drop('Id', 1).fillna(0)
# The number of sets the data needs to split. One for each level
kFoldSplit = 3
# split data into X and y
X = df_Dummy[df_Dummy.columns.difference(["SalePrice"])]
Y = df_Dummy['SalePrice']

col_to_add = np.setdiff1d(X.columns, df_test.columns)

# add these columns to test, setting them equal to zero
for c in col_to_add:
    df_test[c] = 0

firstLevelModels = []
# Models used for regression


selectedModel = modelSelector(X, Y)
firstLevelModels = modelCreator(selectedModel.transform(X), Y)


Y = applyModels(firstLevelModels, selectedModel.transform(df_test))
df_final = pd.DataFrame()
df_final['SalePrice'] = pd.Series(Y)
df_final.index = df_test_id
df_final.to_csv('Output.csv')
