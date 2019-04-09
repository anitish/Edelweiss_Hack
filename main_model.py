import numpy as np
import pandas as pd
import os
import time


# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# pd.set_option('display.expand_frame_repr', True) 


# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import numpy as np
import random
import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, classification_report
import scipy.stats as stats
from sklearn.ensemble import ExtraTreesClassifier

train = pd.read_csv('./data/train_foreclosure.csv')
test = pd.read_csv('./data/test_foreclosure.csv')
customers = pd.read_excel('./data/Customers_31JAN2019.xlsx')
query = pd.read_excel('./data/RF_Final_Data.xlsx')
print('basic data loading done!')

agreements = pd.read_excel('./data/LMS_31JAN2019.xlsx')
print('agreements loaded!!!!')


train_data = train.merge(agreements,on = 'AGREEMENTID',how= 'left').drop_duplicates()
test_data = test.merge(agreements,on = 'AGREEMENTID',how= 'left').drop_duplicates()

print('merged!')


categorical = train_data.select_dtypes(include=[np.object])
categorical = categorical.columns
numeric = train_data.select_dtypes(include=[np.int,np.float])
numeric = numeric.columns
datetime = train_data.select_dtypes(include=[np.datetime64])
datetime = datetime.columns



# #### basic cleaning only
# 

# 1. include this small dummy categories into its parent categories
# 1. normalize skewed numeric variables,(EDA distributions)
# 1. Datetime variables to be dropped

from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()

# Normalization
def norm(data):
    #### variables for minmax
    skewed = ['LOAN_AMT','NET_DISBURSED_AMT','PRE_EMI_DUEAMT', 'PRE_EMI_RECEIVED_AMT',
        'PRE_EMI_OS_AMOUNT','EMI_DUEAMT', 'EMI_RECEIVED_AMT', 'EMI_OS_AMOUNT',
        'EXCESS_AVAILABLE', 'EXCESS_ADJUSTED_AMT', 'BALANCE_EXCESS',
        'NET_RECEIVABLE', 'OUTSTANDING_PRINCIPAL', 'PAID_PRINCIPAL',
        'PAID_INTEREST', 'MONTHOPENING', 'LAST_RECEIPT_AMOUNT']
    data[skewed] = minmax.fit_transform(data[skewed])

norm(train_data)
norm(test_data)


train_data['NPA_IN_LAST_MONTH'].replace(['yes','Yes'],['YES','YES'],inplace = True)
train_data['NPA_IN_CURRENT_MONTH'].replace(['yes','Yes'],['YES','YES'],inplace = True)
test_data['NPA_IN_LAST_MONTH'].replace(['yes','Yes'],['YES','YES'],inplace = True)
test_data['NPA_IN_CURRENT_MONTH'].replace(['yes','Yes'],['YES','YES'],inplace = True)


# corr = train_data.corr()
# corr['FORECLOSURE']


# ### Feature Engineering
train_data['CURRENT_TENOR'].fillna(train_data['ORIGNAL_TENOR'],inplace = True)
train_data['LAST_RECEIPT_AMOUNT'].fillna(0,inplace = True)# as its not paid on that date
train_data['DPD'].fillna(0,inplace = True)# as the tenure hasn't even started for the NaN cases
train_data['BALANCE_TENURE'].fillna(train_data['ORIGNAL_TENOR'],inplace = True)

test_data['CURRENT_TENOR'].fillna(test_data['ORIGNAL_TENOR'],inplace = True)
test_data['LAST_RECEIPT_AMOUNT'].fillna(0,inplace = True)# as its not paid on that date
test_data['DPD'].fillna(0,inplace = True)# as the tenure hasn't even started for the NaN cases
test_data['BALANCE_TENURE'].fillna(test_data['ORIGNAL_TENOR'],inplace = True)



# test_data[test_data['MOB'].isin(train_data['MOB'].unique())]['MOB'].nunique()
## so all the MOB codes are the same in train and test
# lets check if a MOB code signifies only one label


## almost all the codes have same distribution of labels, lets try including the label into features,
## if doesnt turn out good then will remove, rn we will use mean encoding, try label encoding as well


### dropping a few features, keeping PRODUCT, NPA_IN_LAST_MONTH
drop = ['CUSTOMERID','INTEREST_START_DATE','AUTHORIZATIONDATE','CITY','NPA_IN_LAST_MONTH','NPA_IN_CURRENT_MONTH','LAST_RECEIPT_DATE', 'SCHEMEID']
train_new = train_data.drop(drop,axis = 1)
test_new = test_data.drop(drop,axis = 1)


global means, means1#, means2


means = train_new.groupby(['PRODUCT']).FORECLOSURE.mean()
# means1 =  train_new.groupby(['MOB']).FORECLOSURE.mean()
# means2 =   train_new.groupby(['NPA_IN_CURRENT_MONTH']).FORECLOSURE.mean()


def map(data):
    data['PRODUCT'] = data['PRODUCT'].map(means)
#     data['MOB'] = data['MOB'].map(means1)
#     data['NPA_IN_CURRENT_MONTH'] = data['NPA_IN_CURRENT_MONTH'].map(means2)


map(train_new)

map(test_new)


# # ### Model

# X_train, X_valid, y_train, y_valid = train_test_split(train_new.drop(['FORECLOSURE'],axis = 1),
#                                                       train_new['FORECLOSURE'],test_size = 0.2,random_state = 42)

# x_t = X_train.drop(['AGREEMENTID'],axis = 1)
# # x_t = x_t.values
# # y_train = y_train.values
# # y_train = y_train[:,0]
# print('x_t', x_t)
# print('y_train', y_train)


# x_v = X_valid.drop(['AGREEMENTID'],axis = 1)
# # x_v = x_v.values
# # y_valid = y_valid.values
# # y_valid = y_valid[:,0]
# print('x_v',x_v)
# print('y_valid', y_valid)

# t1 = time.time()
# print('TPOT...!')
# tpot = TPOTClassifier(
#     max_time_mins=60 * 10,
#     population_size=100,
#     scoring='roc_auc',
#     cv=3,
#     verbosity=2,
#     random_state=67, n_jobs= -1)
# tpot.fit(x_t, y_train)
# tpot.export('./tpot_pipeline.py')
# print('accuracy is {}'.format(tpot.score(x_v, y_valid)))

# probab = tpot.predict_proba(x_v)
# probab = probab[:,1]
# print('AUC Score is {}'.format(roc_auc_score(y_valid,probab)))
# t2 = time.time()
# print('Total time taken by TPOT:', int(t2-t1))



# check_x = x_v.set_index(X_valid['AGREEMENTID'])

# check_x.set_index(X_valid['AGREEMENTID'],inplace = True)

# check_y = pd.DataFrame(y_valid).set_index(X_valid['AGREEMENTID'])

# check_pred = pd.DataFrame(tpot.predict(x_v)).set_index(X_valid['AGREEMENTID'])

# check_probab = pd.DataFrame(tpot.predict_proba(x_v)).set_index(X_valid['AGREEMENTID'])

# # new_y = check_y.reset_index().groupby(['AGREEMENTID'])['FORECLOSURE'].agg({'y':np.mean})
# new_y = check_y.reset_index().groupby(['AGREEMENTID'])['FORECLOSURE'].agg(lambda x: stats.mode(x)[0][0])

# # new_pred = check_pred.reset_index().groupby(['AGREEMENTID'])[0].agg({'y':stats.mode(axis = None)})
# new_pred = check_pred.reset_index().groupby(['AGREEMENTID'])[0].agg(lambda x: stats.mode(x)[0][0])

# new_probab = check_probab.reset_index().groupby(['AGREEMENTID'])[1].agg({'probab':np.mean})

# print('new_accuracy is {}'.format(np.mean(new_pred==new_y)))

# print('new roc auc is {}'.format(roc_auc_score(new_y,new_probab)))



# print(confusion_matrix(new_y,new_pred))
# print(classification_report(new_y,new_pred))


# ### 1st Submission


X = train_new.drop(['FORECLOSURE','AGREEMENTID'],axis = 1)
y = train_new['FORECLOSURE']


# model_sub =  XGBClassifier(max_depth=10,min_child_weight=5)
# model_sub.fit(X,y)
# model_sub.score(X,y)


# Average CV score on the training set was:0.9996128644226615, X_train,y_train
exported_pipeline = ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.6500000000000001, 
min_samples_leaf=1, min_samples_split=16, n_estimators=100)
model_sub = exported_pipeline
model_sub.fit(X,y)
print('training score is {}'.format(model_sub.score(X,y)))
# exported_pipeline.fit(training_features, training_target)
# results = exported_pipeline.predict(testing_features)


x_t = test_new.drop(['AGREEMENTID','FORECLOSURE'],axis = 1)
check_pred_test = pd.DataFrame(model_sub.predict(x_t)).set_index(test_new['AGREEMENTID'])
check_probab_test = pd.DataFrame(model_sub.predict_proba(x_t)).set_index(test_new['AGREEMENTID'])
new_pred_test = check_pred_test.reset_index().groupby(['AGREEMENTID'])[0].agg(lambda x: stats.mode(x)[0][0])
new_probab_test = check_probab_test.reset_index().groupby(['AGREEMENTID'])[1].agg({'probab':np.mean})


new_probab_test.reset_index(inplace = True)

new_probab_test.rename(columns={'probab':'FORECLOSURE'},inplace = True)

new_probab_test.to_csv('6th_submission.csv',index = False,header=True)
## tpot pipeline model submission. #6th submission