
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import xgboost as xgb
#import matplotlib.pyplot as plt

df_train = pd.read_csv("./train.csv/train.csv", parse_dates=['timestamp'])
df_test = pd.read_csv("./test.csv/test.csv", parse_dates=['timestamp'])
df_macro = pd.read_csv("./macro.csv", parse_dates=['timestamp'])


# =============================
# =============================
# cleanup
# brings error down a lot by removing extreme price per sqm
print(df_train.shape)
df_train.loc[df_train.full_sq == 0, 'full_sq'] = 30
df_train = df_train[df_train.price_doc/df_train.full_sq <= 600000]
df_train = df_train[df_train.price_doc/df_train.full_sq >= 10000]
print(df_train.shape)
#print df_train.describe()
# =============================
# =============================

y_train = df_train['price_doc'].values
id_test = df_test['id']

df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
df_test.drop(['id'], axis=1, inplace=True)

# Build df_all = (df_train+df_test).join(df_macro)
num_train = len(df_train)
df_all = pd.concat([df_train, df_test])
print df_all.shape
print df_all.columns
df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')
print(df_all.shape)
print df_all.columns
# ==============================

# Add month-year
month_year = (df_all.timestamp.dt.month + df_all.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
df_all['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (df_all.timestamp.dt.weekofyear + df_all.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
df_all['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
df_all['month'] = df_all.timestamp.dt.month
df_all['dow'] = df_all.timestamp.dt.dayofweek

# Other feature engineering
df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)
#df_all["FAR"]=df_all["full_sq"]/df_all["area_m"].astype(float)
df_all["rest_area"]=(df_all["full_sq"]-df_all["kitch_sq"]).astype(float)
df_all["inverse_floor"] = df_all["max_floor"]-df_all["floor"]
#df_all["sqrt_area"] = np.sqrt(df_all["area_m"])
#print df_all.loc[:,["sub_area","full_sq","life_sq","sqrt_area","area_m"]]

#df_all["st_mate"] = df_all["state"]*df_all["material"]

col = "build_year"
df_all[col].fillna(0,inplace=True)
ulimit = np.percentile(df_all[col].values, 99.995)
#llimit = np.percentile(df_all[col].values, 0.005)
df_all[col].ix[df_all[col]>ulimit] = ulimit
#df_all[col].ix[df_all[col]<llimit] = llimit

#df_all["build_year"].describe()

#print df_all.loc[:,["build_count_before_1920","build_count_1921-1945","build_count_1946-1970","build_count_1971-1995","build_count_after_1995"]]

df_all.loc[:,["timestamp","sub_area","raion_build_count_with_builddate_info","build_count_before_1920","build_count_1921-1945",
              "build_count_1946-1970","build_count_1971-1995","build_count_after_1995","build_year"]]

df_all["room_area"] = ((df_all["life_sq"] - df_all["kitch_sq"])/df_all["num_room"]).astype(float)

df_all.loc[:,["sub_area","raion_popul"]]

#df_all["raion_build_count_with_builddate_info"].describe([0.01,0.99])

# Remove timestamp column (may overfit the model in train)
df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)

factorize = lambda t: pd.factorize(t[1])[0]

df_obj = df_all.select_dtypes(include=['object'])

X_all = np.c_[
    df_all.select_dtypes(exclude=['object']).values,
    np.array(list(map(factorize, df_obj.iteritems()))).T
]
print(X_all.shape)

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
print(X_all.shape)

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

#cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
#    verbose_eval=True, show_stdv=False)
#cv_result[['train-rmse-mean', 'test-rmse-mean']].plot()
#num_boost_rounds = len(cv_result)

num_boost_round = 489

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_round)

#fig, ax = plt.subplots(1, 1, figsize=(8, 16))
#xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

y_pred = model.predict(dtest)
y_pred = np.round(y_pred * 0.99)
df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})

df_sub.to_csv('sub_525_2.csv', index=False)

