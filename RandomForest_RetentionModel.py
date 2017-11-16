
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
filename = "/home/lovedeep/Desktop/HUB/lovedeep_data.xlsx"
df_orig = pd.read_excel(filename)


# In[2]:


NUMERIC_COLUMNS = df_orig.dtypes[df_orig.dtypes == 'float64'].index
print(NUMERIC_COLUMNS)


# In[3]:


print(df_orig.shape)
print(df.shape)
print(df.PolicyStatusName.describe())


# In[5]:


#df[LABELS].apply(pd.Series.nunique).plot(kind='pie', figsize=(6,3))
fig, ax = plt.subplots(1, 1)
ax.pie(df.PolicyStatusName.value_counts(),autopct='%1.1f%%', labels=['Renewed','NonRenewed'], colors=['yellowgreen','r'])
plt.axis('equal')
plt.ylabel('')
plt.show()


# In[6]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df["Class"] = le.fit_transform(df.PolicyStatusName)
df_orig["Class"] = le.fit_transform(df_orig.PolicyStatusName)
#print(df.PolicyStatusName)
df['Class'].value_counts().plot(kind="bar", figsize=(10,3))
plt.show()


# In[7]:


drop_list = ["ClientName", "PostalCode", "PolicyNumber", "DepartmentName", "DepartmentCode", "CommissionTypeCode",
             'PolicyEffectiveDate', 'PolicyExpirationDate','ContractedExpirationDate',
            'City', 'State', 'PolicyStatusName'
            ]
df = df.drop(drop_list, axis=1)
df_orig = df_orig.drop(drop_list, axis=1)


# In[8]:


df.shape


# In[9]:


df_orig.shape


# In[10]:


LABELS = ['Class']
NON_LABELS = [c for c in df.columns if c not in LABELS]
print(NON_LABELS)


# In[11]:


main_NUMERIC_COLUMNS = ['BilledPremiumAmt', 'BilledCommissionAmt', 'AnnualizedPremiumAmt', 'AnnualizedCommissionAmt', 
                        'EstimatedPremiumAmt', 'EstimatedCommissionAmt', 'AccrualAmt']
keep = list(NUMERIC_COLUMNS) + list(LABELS)
df = df[keep]
df.shape
print(df.dtypes)


# In[12]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer


# In[13]:


# split data for training and testing
def split_data(df, drop_list, test_size):
    df = df.drop(drop_list,axis=1)
    print(df.columns)
    #test train split time
    from sklearn.model_selection import train_test_split
    y = df['Class'].values #target
    X = df.drop(['Class'],axis=1).values #features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                    random_state=42, stratify=y)

    print("train-set size: ", len(y_train),
      "\ntest-set size: ", len(y_test))
    print("positive (renewed) cases in test-set: ", sum(y_test))
    return X_train, X_test, y_train, y_test


# In[14]:


drop_list = []
X_train, X_test, y_train, y_test = split_data(df, drop_list, 0.2)


# In[15]:


rf = RandomForestClassifier(n_estimators = 400, max_depth = 4, 
                            n_jobs=-1, class_weight='balanced')
rf.fit(X_train, y_train)


# In[16]:


def gini(truth, predictions):
    g = np.asarray(np.c_[truth, predictions, np.arange(len(truth)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(truth) + 1) / 2.
    return gs / len(truth)

def gini_xgb(predictions, truth):
    truth = truth.get_label()
    return 'gini', -1.0 * gini(truth, predictions) / gini(truth, truth)

def gini_lgb(truth, predictions):
    score = gini(truth, predictions) / gini(truth, truth)
    return 'gini', score, True

def gini_sklearn(truth, predictions):
    return gini(truth, predictions) / gini(truth, truth)

gini_scorer = make_scorer(gini_sklearn, greater_is_better=True, needs_proba=True)
#define the said metric: gini score


# In[17]:


# get the predictions
def get_predictions(clf, X_test):
    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = clf.predict_proba(X_test)
    y_pred_prob = y_pred_prob[:,1]
    return y_pred_prob


# In[18]:


# print gini score
def print_scores(y_test, y_pred_prob):
    gini_predictions = gini(y_test, y_pred_prob)
    gini_max = gini(y_test, y_test)
    ngini= gini_sklearn(y_test, y_pred_prob)
    print('Normalized Gini: %.3f' 
      % (ngini))


# In[19]:


from sklearn.metrics import confusion_matrix,auc,roc_auc_score
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score


# In[20]:


def print_confusionmatrix(y_test,y_pred,y_pred_prob):
    print('test-set confusion matrix:\n', confusion_matrix(y_test,y_test)) 
    print('test-set confusion matrix:\n', confusion_matrix(y_test,y_pred)) 
    print("recall score: ", recall_score(y_test,y_pred))
    print("precision score: ", precision_score(y_test,y_pred))
    print("f1 score: ", f1_score(y_test,y_pred))
    print("accuracy score: ", accuracy_score(y_test,y_pred))
    print("ROC AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))


# In[21]:


y_pred_prob = get_predictions(rf, X_test)


# In[22]:


y_pred = rf.predict(X_test)


# In[26]:


print_scores(y_test, y_pred_prob)


# In[24]:


print_confusionmatrix(y_test,y_pred,y_pred_prob)


# In[27]:


# Format predictions in DataFrame: prediction_df
prediction_df = pd.DataFrame(columns=df[LABELS].columns,
                             data=y_pred_prob)

prediction_df["known_label"] = y_test
# Save prediction_df to csv
prediction_df.to_csv('num_predictions.csv')

