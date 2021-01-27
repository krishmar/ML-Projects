import numpy as np
import scipy.stats
import scipy.special
import pandas as pd
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.ensemble


#********************************************************************************
#   Fetch Data
#********************************************************************************

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
#train_df.head(3)

#********************************************************************************
#   Data Wrangling
#********************************************************************************

train_df['Log1pSalePrice'] = np.log1p(train_df["SalePrice"])
data_df = pd.concat((train_df, test_df)).reset_index(drop=True)
data_df.drop(['SalePrice','Log1pSalePrice'],axis=1,inplace=True)
# data_df.shape  

#****************************************
#****************************************
# fillna('None')

fill_none_col = ['PoolQC','MiscFeature',
           'Alley','Fence', 
           'FireplaceQu','GarageType',
           'GarageFinish', 'GarageQual',
           'GarageCond','BsmtQual',
           'BsmtCond', 'BsmtExposure',
           'BsmtFinType1','BsmtFinType2',
           'MasVnrType', 'MSSubClass']

for col in fill_none_col:
    data_df[col] = data_df[col].fillna('None')
    
#****************************************
#****************************************
# fillna(0)

fill_0_col = ['GarageYrBlt', 'GarageArea',
           'GarageCars','BsmtFinSF1',
           'BsmtFinSF2', 'BsmtUnfSF',
           'TotalBsmtSF', 'BsmtFullBath', 
            'BsmtHalfBath',"MasVnrArea"]

for col in fill_0_col:
    data_df[col] = data_df[col].fillna(0)
    
#****************************************
#****************************************
# fillna(mode())

fill_mode_col = ['Electrical','KitchenQual',
           'KitchenQual','Exterior2nd',
           'SaleType']

for col in fill_mode_col:
    data_df[col] = data_df[col].fillna(data_df[col].mode()[0])
    
#****************************************
#****************************************


data_df = data_df.drop(['Utilities'], axis=1)

data_df['LotFrontage'] = data_df.groupby('Neighborhood')['LotFrontage'].transform(
						lambda x: x.fillna(x.median()))

data_df["Functional"] = data_df["Functional"].fillna("Typ")

# data_df_na = data_df.isnull().median(axis =0)
# data_df_na = data_df_na.drop(data_df_na[data_df_na == 0].index).sort_values(ascending = False)
# missing_data = pd.DataFrame({'Missing Data Ratio': data_df_na})
# print('data_df_na.shape = ', data_df_na.shape)
# missing_data.head()

#****************************************
#****************************************

conv_str_col = ['MSSubClass','OverallCond','YrSold','MoSold']

for col in conv_str_col:
    data_df[col] = data_df[col].astype(str)

encode_col = ('FireplaceQu', 'BsmtQual', 
        'BsmtCond', 'GarageQual', 
        'GarageCond', 'ExterQual',
        'ExterCond','HeatingQC',
        'PoolQC', 'KitchenQual',
        'BsmtFinType1','BsmtFinType2',
        'Functional', 'Fence',
        'BsmtExposure', 'GarageFinish',
        'LandSlope','LotShape',
        'PavedDrive', 'Street',
        'Alley', 'CentralAir',
        'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

for c in encode_col:
    lbl = sklearn.preprocessing.LabelEncoder()
    lbl.fit(list(data_df[c].values))
    data_df[c] = lbl.transform(list(data_df[c].values))

data_df['TotalSF'] = data_df['TotalBsmtSF'] + data_df['1stFlrSF']+ data_df['2ndFlrSF']

numerical_features = data_df.dtypes[data_df.dtypes != 'object'].index
skewness_of_features = data_df[numerical_features].apply(
	lambda x: scipy.stats.skew(x.dropna())).sort_values(ascending=False)
skewness_df = pd.DataFrame({'Skewness': skewness_of_features})
skewness_df.head(3)

skewness_df  = skewness_df[abs(skewness_df) > 0.75]
skewness_features = skewness_df.index
lamb  = 0.15
for features in skewness_features:
    data_df[features] = scipy.special.boxcox1p(data_df[features],lamb)

df = data_df[:train_df.shape[0]]
df['SalePrice'] = train_df['SalePrice'].values

data_df = pd.get_dummies(data_df).copy()

#********************************************************************************
#   Build, Train, Test model
#********************************************************************************


def get_rmse(y_pred, y_target):
    return np.sqrt(np.mean(np.square(y_pred.reshape(-1,) - y_target.reshape(-1,))))

def normalize_data(data):
    rs = sklearn.preprocessing.RobustScaler()
    rs.fit(data)
    data = rs.transform(data)
    return data

data_df_norm = normalize_data(data_df.values)

x_train_valid = data_df_norm[:train_df.shape[0]]
y_train_valid = train_df.Log1pSalePrice.values
x_test = data_df_norm[train_df.shape[0]:]

rmse_train = {}
rmse_valid = {}
y_train_pred = {}
y_valid_pred = {}
y_test_pred = {}

lasso = sklearn.linear_model.Lasso(alpha = 0.0005, random_state=1)
#lasso = sklearn.linear_model.Lasso(alpha = 0.0005)

gboost = sklearn.ensemble.GradientBoostingRegressor(n_estimators=3000,
            learning_rate=0.05, max_features ='sqrt', min_samples_leaf=15,
            min_samples_split=10, loss='huber', random_state =5)

model_init = {
    'lasso':lasso,
    'gboost':gboost
}

take_base_models = ['lasso', 'gboost', 'lasso',
                    'gboost', 'gboost','lasso', 
                    'gboost', 'lasso', 'gboost',
                    'gboost']

take_meta_model = 'lasso'

kfold = sklearn.model_selection.KFold(len(take_base_models),shuffle = True)

x_train_meta = np.array([])
y_train_meta = np.array([])
x_test_meta = np.zeros(x_test.shape[0])

for i, (train_index, valid_index) in enumerate(kfold.split(x_train_valid)):
        x_train = x_train_valid[train_index]
        y_train = y_train_valid[train_index]
        x_valid = x_train_valid[valid_index]
        y_valid = y_train_valid[valid_index]
        
        model = sklearn.base.clone(model_init[take_base_models[i]])
        model.fit(x_train, y_train)
        y_train_pred['tmp'] = model.predict(x_train)
        y_valid_pred['tmp'] = model.predict(x_valid)
        y_test_pred['tmp'] = model.predict(x_test)
        
        x_train_meta = np.concatenate([x_train_meta, y_valid_pred['tmp']])
        y_train_meta = np.concatenate([y_train_meta, y_valid])
        x_test_meta += y_test_pred['tmp']
        
        
        print(take_base_models[i], ':train/valid rmse = %.3f/%.3f'%(
            get_rmse(y_train_pred['tmp'], y_train), get_rmse(y_valid_pred['tmp'], y_valid)))
        
        
x_train_meta = x_train_meta.reshape(-1,1)
x_test_meta = (x_test_meta / len(take_base_models)).reshape(-1,1)
# y_test_pred['stacked'] = x_test_meta

model = sklearn.base.clone(model_init[take_meta_model])
model.fit(x_train_meta, y_train_meta)
y_train_pred['meta model'] = model.predict(x_train_meta)
y_test_pred['meta model'] = model.predict(x_test_meta)

print('Meta model: train rmse =', get_rmse(x_train_meta, y_train_pred['meta model']))

y_test_submit = y_test_pred['meta model']

sub_df = pd.DataFrame()
sub_df['Id'] = test_df['Id'].values
sub_df['SalePrice'] = np.expm1(y_test_submit)
sub_df.to_csv('submission.csv',index=False)
sub_df.head(5)

print('Submission.csv is created and ready for submission')