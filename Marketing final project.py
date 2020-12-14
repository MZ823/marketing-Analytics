#!/usr/bin/env python
# coding: utf-8

# # Marginal CAC
# 

# In[129]:


import pandas as pd
import numpy as np


# In[255]:


df=pd.read_csv('subscribers.csv')
df.head()


# In[252]:


# Check NA rows
df['account_creation_date'].isna().sum() 


# In[248]:


df = df.dropna(subset=['account_creation_date'])
df.head()


# In[ ]:


#extract account create date
df['account_creation_date']=pd.to_datetime(df['account_creation_date'], format='%Y-%m-%d %H:%M:%S')

df['creation_month'] = pd.DatetimeIndex(df['account_creation_date']).month
df['creation_month'].apply(lambda x: int(x))


# In[72]:


#f.to_csv('month.csv')


# In[84]:


df=pd.read_csv('monthssss.csv')
df.head()


# In[85]:


#extract account create date
df['account_creation_date']=pd.to_datetime(df['account_creation_date'], format='%Y-%m-%d %H:%M')
df['creation_month'] = pd.DatetimeIndex(df['account_creation_date']).month
df['creation_month'].apply(lambda x: int(x))


# In[87]:


df.to_csv('monthssss.csv')


# # Churn Model

# In[57]:


import pandas as pd


# In[74]:


df=pd.read_csv('subscribers.csv')
df.head()


# In[75]:


df['num_weekly_services_utilized'].isna().sum() 
df = df.dropna(subset=['age', 'male_TF','package_type','intended_use','weekly_consumption_hour','current_sub_TF','op_sys'])
#Change True and False to 0 and 1
df.replace(False, str(0), inplace=True)
df.replace(True, str(1), inplace=True)
df


# In[76]:


df.to_csv('churn_need_to_delete_age.csv')


# In[ ]:





# In[428]:


df=pd.read_csv('churn_age_deleted.csv')


# In[429]:


df = df.drop(['Unnamed: 0','Unnamed: 0.1',
              'country','subid','language',
              'payment_type', 'num_trial_days',
              'last_payment','next_payment',
              'cancel_date', 'months_per_bill_period', 
              'account_creation_date','trial_end_date',
              'initial_credit_card_declined','plan_type',
              'current_sub_TF',
              'discount_price','op_sys','payment_period'],axis=1)
df.head()


# In[430]:


list(df.columns) 


# In[431]:


df = pd.get_dummies(df, columns=['package_type', 'preferred_genre',
                                 'intended_use','attribution_technical',
                                 'attribution_survey','monthly_price','join_fee'])


# In[432]:


df = df.dropna(subset=['num_weekly_services_utilized','num_ideal_streaming_services'])


# In[406]:


df


# In[387]:


#df.to_csv('churn.csv')


# In[407]:


df.isna().sum() 


# In[408]:


cor = df.corr()
cor


# In[44]:


#Using Pearson Correlation
plt.figure(figsize=(24,20))
sns.heatmap(cor)
plt.show()


# In[409]:


#cor.to_csv('corr.csv')


# In[423]:


coeff = cor

# 0.3 is used for illustration 
# replace with your actual value
thresh = 0.7

mask = coeff.abs()> thresh
# or mask = coeff < thresh

coeff.where(mask).stack()


# In[421]:


high_corr


# In[378]:


#Correlation with output variable
cor_target = abs(cor["churn"])
cor_target

#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.4]
relevant_features


# In[317]:





# ## building the churn model

# ### logistic regression model

# In[517]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import sklearn.metrics as metrics
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[528]:


df=pd.read_csv('churn.csv')
df


# In[519]:


# logistic Model
feature_x = [tag for tag in df.columns if tag not in ['churn','payment_period','trial_completed ']]
X = df[feature_x].values
y = df['churn']


# In[520]:


# training and testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#running the model
logistic_regression= LogisticRegression(max_iter=200000)
logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)

probs = logistic_regression.predict_proba(X_test)
preds = probs[:,1]

fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.figure(figsize=(100,100))
plt.show()


# In[532]:


feature_x = [tag for tag in df.columns if tag not in ['churn','payment_period','trial_completed ']]
X = df[feature_x].values
y = df['churn']


# In[535]:


logistic_regression= LogisticRegression(max_iter=200000)
logistic_regression.fit(X,y)
y_pred=logistic_regression.predict(X)
probs = logistic_regression.predict_proba(X)
preds = probs[:,1]


# In[536]:


df['probs']= preds


# In[538]:


df.to_csv('churn_with_pred.csv')


# import statsmodels.api as sm
# logit_model=sm.Logit(y,X)
# result=logit_model.fit()
# print(result.summary())

# ### decision tree model

# In[56]:


from sklearn.tree import DecisionTreeClassifier


# In[54]:


df=pd.read_csv('churn.csv')
df


# In[59]:


feature_x = [tag for tag in df.columns if tag not in ['churn','payment_period','Unnamed: 0']]
X = df[feature_x].values
y = df['churn']
# training and testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

probs = clf.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.figure(figsize=(100,100))
plt.show()


# ### Random Forest Model

# In[61]:


from sklearn.ensemble import RandomForestClassifier


# In[148]:


df= pd.read_csv('churn.csv')
df


# In[149]:


feature_x = [tag for tag in df.columns if tag not in ['churn','payment_period','trial_completed','Unnamed: 0']]
X = df[feature_x].values
y = df['churn']
# training and testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
clf = RandomForestClassifier(n_estimators=100)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[152]:


probs = clf.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.figure(figsize=(100,100))
plt.show()


# In[ ]:





# # Customer Segmentation

# In[366]:


import pandas as pd


# In[487]:


df=pd.read_csv('churn_age_deleted.csv')


# In[488]:


df= df.drop(['Unnamed: 0','Unnamed: 0.1',
              'last_payment',
              'next_payment','language',
              'payment_type','num_trial_days', 'trial_end_date','plan_type',
              'months_per_bill_period', 'country','monthly_price','discount_price','trial_completed'],axis=1)


# In[489]:


list(df.columns)


# In[490]:


df.cancel_date = pd.notna(df.cancel_date)
df.replace(False, str(0), inplace=True)
df.replace(True, str(1), inplace=True)

#extract account create date
df['account_creation_date']=pd.to_datetime(df['account_creation_date'], format='%Y-%m-%d %H:%M')
df['account_creation_month'] = pd.DatetimeIndex(df['account_creation_date']).month
df['account_creation_hour'] = pd.DatetimeIndex(df['account_creation_date']).hour
df= df.drop(['account_creation_date'],axis=1)
df.head()


# In[491]:


#df=pd.read_csv('churn_age_deleted.csv')
df.to_csv('data_before_cluster.csv')


# In[480]:


df = pd.get_dummies(df, columns=['package_type','num_weekly_services_utilized', 
                                 'preferred_genre','intended_use','num_ideal_streaming_services',
                                 'attribution_technical','attribution_survey','op_sys',
                                 'join_fee'],drop_first=True)
df.head()


# In[481]:


X =list(df.columns) 
#feature_x = [tag for tag in df.columns if tag not in ['subid']]
#x = df[feature_x].values

import pandas as pd
from sklearn import preprocessing
x = df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

df.columns = X


dk=pd.read_csv('churn_age_deleted.csv')
df['subid']=dk['subid']
df


# In[482]:


feature_x = [tag for tag in df.columns if tag not in ['subid']]
X = df[feature_x].values


# In[483]:


import matplotlib.pyplot as plt
from sklearn. cluster import KMeans

inertias = {}
for k in range(1,20):
    kmeans = KMeans(n_clusters=k, random_state=2020)
    kmeans.fit(df)
    inertias[k] = kmeans.inertia_
print(inertias)


ax = plt.subplot()
ax.plot(list(inertias.keys()), list(inertias.values()), '-*')
ax.set_xticks(np.arange(1, 20))
ax.grid()
plt.show()


# In[484]:


k = 4
kmeans = KMeans(n_clusters=k, random_state=2020)
y_pred = kmeans.fit_predict(X)
df['cluster']=y_pred


# In[486]:


df.to_csv('data_cluster.csv')


# In[ ]:





# In[386]:


def visualize_cluster_result(x, dim1, dim2, y_pred, k):
    # select two feature dims: dim1, dim2, visualize the clusters
    assert dim1 in range(x.shape[1])
    assert dim2 in range(x.shape[1])
    ax = plt.subplot()
    # use a for loop to plot each cluster (with different colors)
    for i in range(k):
        ax.scatter(x[y_pred == i, dim1], x[y_pred ==  i, dim2], label='cluster: %d' % i)
    plt.title('Visualization of clustering of dim %d and dim %d' % (dim1, dim2))
    ax.set_xlabel('dim: %d' % dim1)
    ax.set_ylabel('dim: %d' % dim2)
    ax.legend()
    plt.show()


# In[400]:


visualize_cluster_result(X, 0, 1, y_pred, k)


# In[493]:


df1=pd.read_csv('data_before_cluster.csv')
df2=pd.read_csv('data_cluster.csv')


# In[494]:


df1['cluster']=df2['cluster']


# In[495]:


df1.to_csv('data_cluster_final.csv')


# In[ ]:




