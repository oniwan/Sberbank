
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import xgboost as xgb
#import matplotlib.pyplot as plt
#import seaborn as sns
#%matplotlib inline

df_train = pd.read_csv("./train.csv/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("./test.csv/test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("./macro.csv", parse_dates=['timestamp'])


# In[ ]:


# =============================
# =============================
# cleanup
# brings error down a lot by removing extreme price per sqm
#print(df_train.shape)

#Very Strange data
df_train.loc[df_train['id'] == 10092, 'build_year'] = 2007
df_train.loc[df_train['id'] == 10092, 'state'] = 3
df_train.loc[df_train['id'] == 10093, 'build_year'] = 2009


#Too High and Low
#print df_train[df_train.price_doc / df_train.full_sq > 600000]
#df_train = df_train[df_train.price_doc/df_train.full_sq <= 600000]
#df_train = df_train[df_train.price_doc/df_train.full_sq >= 10000]
#print df_train.shape
# =============================
# =============================


# In[ ]:


#==============================
y_train =df_train['price_doc'].values
y_train[:753] = y_train[:753] * 1.5945
y_train[753:5593] = y_train[753:5593] * 1.503
y_train[5593:13571] = y_train[5593:13571] * 1.4108
y_train[13571:27233] = y_train[13571:27233] * 1.3249
y_train[27233:] = y_train[27233:]* 1.1897
Y = df_train["price_doc"].values
id_test = df_test['id']

#df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
#df_test.drop(['id'], axis=1, inplace=True)
#=============================


# In[ ]:


# Build df_all = (df_train+df_test).join(df_macro)
num_train = len(df_train)
df_all = pd.concat([df_train, df_test])
#df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')
# ==============================
#Clean Strange Data

#full_sq is 0
bad_index = df_all[df_all.full_sq == 0].index
df_all.ix[bad_index, 'full_sq'] = np.NaN

#Life_sq > Full_sq is strange
bad_index = df_all[df_all.life_sq > df_all.full_sq].index
df_all.ix[bad_index, "life_sq"] = np.NaN

#Remove too small Life_Sq data 
bad_index = df_all[df_all.life_sq < 5].index
df_all.ix[bad_index, "life_sq"] = np.NaN

#Remove too small Full_sq data
bad_index = df_all[df_all.full_sq < 5].index
df_all.ix[bad_index, "full_sq"] = np.NaN

#
kitch_is_build_year = [13117]
df_all.ix[kitch_is_build_year, "build_year"] = df_all.ix[kitch_is_build_year, "kitch_sq"]

#Kitch_sq > life_sq is strange
bad_index = df_all[df_all.kitch_sq >= df_all.life_sq].index
df_all.ix[bad_index, "kitch_sq"] = np.NaN

#Kitch_sq == 0 or ==1 is strange
bad_index = df_all[(df_all.kitch_sq == 0).values + (df_all.kitch_sq == 1).values].index
df_all.ix[bad_index, "kitch_sq"] = np.NaN

#
bad_index = df_all[(df_all.full_sq > 210) & (df_all.life_sq / df_all.full_sq < 0.3)].index
df_all.ix[bad_index, "full_sq"] = np.NaN

#Too old house 
bad_index = df_all[df_all.build_year < 1500].index
df_all.ix[bad_index, "build_year"] = np.NaN

#Too new house
bad_index = df_all[df_all.build_year > 2020].index
df_all.ix[bad_index, "build_year"] = np.NaN

#No room 
bad_index = df_all[df_all.num_room == 0].index 
df_all.ix[bad_index, "num_room"] = np.NaN

#No max floor and floor
bad_index = df_all[(df_all.floor == 0).values * (df_all.max_floor == 0).values].index
df_all.ix[bad_index, ["max_floor", "floor"]] = np.NaN

#No floor
bad_index = df_all[df_all.floor == 0].index
df_all.ix[bad_index, "floor"] = np.NaN

#No max floor
bad_index = df_all[df_all.max_floor == 0].index
df_all.ix[bad_index, "max_floor"]=np.NaN

#floor > max_floor 
bad_index = df_all[df_all.floor > df_all.max_floor].index
df_all.ix[bad_index, "max_floor"] = np.NaN

#23584 is strange
bad_index = [23584]
df_all.ix[bad_index, "floor"] = np.NaN

#1 <= state <= 4 
bad_index = df_all[df_all.state == 33].index
df_all.ix[bad_index, "state"] = np.NaN



# In[ ]:


# Add month-year
#month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
#month_year_cnt_map = month_year.value_counts().to_dict()
#df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
#week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
#week_year_cnt_map = week_year.value_counts().to_dict()
#df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
#df_all['month'] = df_all.timestamp.dt.month
#df_all['dow'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)
df_all["inverse_floor"] = df_all["max_floor"]-df_all["floor"]
df_all["extra_area"] = df_all["full_sq"] - df_all["life_sq"]
df_all["room_size"] = (df_all["life_sq"] - df_all["kitch_sq"])/df_all["num_room"]
#df_all["pop_density_raion"] = df_all["raion_popul"]/df_all["area_m"]
#df_all["young_proportion"] =df_all["young_all"] / df_all["full_all"]
#df_all["work_proportion"] = df_all["work_all"] / df_all["full_all"]
#df_all["retire_proportion"] = df_all["ekder_all"] / df_all["full_all"]
#df_all["ratio_preschool"] = df_all["children_school"] /df_all["preschool_quota"]
#df_all["ratio_school"] = df_all["children_school"] / df_all["school_quota"]
#df_all["year_old"] = df_all.timestamp.dt.year-df_all["build_year"]
df_all["apartmentname"] =df_all.sub_area + df_all['metro_km_avto'].astype(str)


#df_all["FAR"]=df_all["full_sq"]/df_all["area_m"].astype(float)
#df_all["ratio_life"]=((df_all["full_sq"]-df_all["kitch_sq"])/df_all["life_sq"]).astype(float)
#df_all["sqrt_area"] = np.sqrt(df_all["area_m"])
#df_all["log_full_sq"] = np.log(df_all["full_sq"])
#df_all["log_life_sq"] = np.log(df_all["life_sq"])
#df_all["log_kitch_sq"] = np.log(df_all["kitch_sq"])
#df_all["log_area_m"] = np.log(df_all["area_m"])
#df_all.drop(['full_sq','life_sq','kitch_sq','area_m'],axis=1,inplace=True)
#print df_all.loc[:,["sub_area","full_sq","life_sq","sqrt_area","area_m"]]


# In[ ]:


df_all.drop(['id', 'price_doc'], axis=1, inplace=True)

# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp'], axis=1, inplace=True)

factorize = lambda t: pd.factorize(t[1])[0]

df_obj = df_all.select_dtypes(include=['object'])

X_all = np.c_[
    df_all.select_dtypes(exclude=['object']).values,
    np.array(list(map(factorize, df_obj.iteritems()))).T
]

#print(X_all.shape)

X_train = X_all[:num_train]
X_test = X_all[num_train:]


# Deal with categorical values
df_numeric = df_all.select_dtypes(exclude=['object'])
df_obj = df_all.select_dtypes(include=['object']).copy()

for c in df_obj:
    df_obj[c] = pd.factorize(df_obj[c])[0]

df_values = pd.concat([df_numeric, df_obj], axis=1)


# Convert to numpy values
X_all = df_values.values
X_train = X_all[:num_train]
X_test = X_all[num_train:]

df_columns = df_values.columns


xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
dtest = xgb.DMatrix(X_test, feature_names=df_columns)


# Uncomment to tune XGB `num_boost_rounds`

cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=2000, early_stopping_rounds=20,verbose_eval=True, show_stdv=False)
#cv_result[['train-rmse-mean', 'test-rmse-mean']].plot()
print cv_result['test-rmse-mean'].min()


# In[ ]:


num_boost_round = len(cv_result)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_round)

#fig, ax = plt.subplots(1, 1, figsize=(15, 50))
#xgb.plot_importance(model,height=0.5, ax=ax)
#plt.show()

y_pred = model.predict(dtest)
y_pred = np.round(y_pred)
y_pred[:3679] = y_pred[:3679] * 0.848
y_pred[3679:] = y_pred[3679:] * 0.949
df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})


df_sub.to_csv('sub_0601_2.csv', index=False)


# In[ ]:


print df_sub


# In[ ]:





# In[ ]:




