import numpy as np
import pandas as pd
import xgboost as xgb
#import matplotlib.pyplot as plt

df_train = pd.read_csv('/home/2829902373/Sberbank/train.csv/train.csv')
df_test = pd.read_csv('/home/2829902373/Sberbank/test.csv/test.csv')
df_macro = pd.read_csv('/home/2829902373/Sberbank/macro.csv')

df_train.loc[df_train['id']==10092,'build_year']=2007
df_train.loc[df_train['id']==10092,'state'] = 3
df_train.loc[df_train['id']==10093,'build_year']=2009

y_train = df_train['price_doc'].values
id_test = df_test['id']

bad_index = df_train[df_train.full_sq == 0].index
df_train.ix[bad_in
