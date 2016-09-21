import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.linear_model import LassoCV
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectFromModel


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                      test.loc[:, 'MSSubClass':'SaleCondition']))

# log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

dataSets = [train, all_data]
for dataset in dataSets:
    # All the categorical columns that have Excellent, Good, Average/Typical,Fair,
    # Poor and NA as it is values
    cols = ['PoolQC', 'GarageCond', 'GarageQual', 'FireplaceQu', 'KitchenQual',
            'BsmtCond', 'BsmtQual', 'ExterCond', 'ExterQual', 'HeatingQC']
    for c in cols:
        dataset[c] = dataset[c].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2,
                                     'Po': 1, 'NA': 0})
    # This is required as the NA in csv is getting read as NAN
        dataset[c] = dataset[c].fillna(0)

    dataset.Fence = dataset.Fence.map({'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2,
                                       'MnWw': 1, 'NA': 0}).fillna(0)
    dataset.GarageFinish = dataset.GarageFinish.map({'Fin': 3, 'RFn': 2,
                                                     'Unf': 1, 'NA': 0}
                                                    ).fillna(0)

    cols = ['BsmtFinType1', 'BsmtFinType2']
    for c in cols:
        dataset[c] = dataset[c].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4,
                                    'Rec': 3, 'LwQ': 2, 'Unf': 1,
                                     'NA': 0}).fillna(0)

    dataset.BsmtExposure = dataset.BsmtExposure.map({'Gd': 4, 'Av': 3, 'Mn': 2,
                                                    'No': 1, ' NA': 0}
                                                    ).fillna(0)

    dataset.MSSubClass = dataset.MSSubClass.map({20:'1-STORY 1946 & NEWER ALL STYLES', 30: '1-STORY 1945 & OLDER'
                                         , 40:'1-STORY W/FINISHED ATTIC ALL AGES', 45: '1-1/2 STORY - UNFINISHED ALL AGES'
                                        , 50:'1-1/2 STORY FINISHED ALL AGES', 60:'2-STORY 1946 & NEWER', 70:'2-STORY 1945 & OLDER',
                                        75:'2-1/2 STORY ALL AGES', 80:'SPLIT OR MULTI-LEVEL', 85:'SPLIT FOYER', 90:'DUPLEX - ALL STYLES AND AGES',
                                        120:'1-STORY PUD (Planned Unit Development) - 1946 & NEWER',
                                        150:'1-1/2 STORY PUD - ALL AGES', 160:'2-STORY PUD - 1946 & NEWER',
                                        180:'PUD - MULTILEVEL - INCL SPLIT LEV/FOYER', 190:'2 FAMILY CONVERSION - ALL STYLES AND AGES'})


# log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# For right skewed data
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.8]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# For left skewed data
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats < -0.8]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.square(all_data[skewed_feats])


all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice


def rmse_cv(model, X):
    rmse = np.sqrt(-cross_val_score(model, X, y,
                   scoring="mean_squared_error", cv=5))
    print rmse


model_lasso = LassoCV(alphas=[1, 0.1, 0.001, 0.0005]).fit(X_train, y)
coefs = model_lasso.coef_
for c in coefs:
    sfm = SelectFromModel(model_lasso, threshold=c, prefit=True)
    print "For Coeff:", c, "The RMSE are", rmse_cv(model_lasso, sfm.transform(X_train))
'''
preds = np.expm1(model_lasso.predict(X_test))
solution = pd.DataFrame({"id": test.Id, "SalePrice": preds})
solution.to_csv("Try2.csv", index=False)
'''
