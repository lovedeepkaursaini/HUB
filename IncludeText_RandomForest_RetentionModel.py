
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
            'City', 'State', 'PolicyStatusName', 'InsurerName', 'BrokerName', 'ProducerName'
            ]
df = df.drop(drop_list, axis=1)
df_orig = df_orig.drop(drop_list, axis=1)


# In[8]:


print(df.dtypes)


# In[9]:


df.shape


# In[10]:


df_orig.shape


# In[11]:


LABELS = ['Class']
NON_LABELS = [c for c in df.columns if c not in LABELS]
print(NON_LABELS)


# In[12]:


# Define combine_text_columns()
def combine_text_columns(df, to_drop= list(NUMERIC_COLUMNS) + list(LABELS)):
    """ converts all text in each row of data_frame to single vector """
    
    # Drop non-text columns that are in the df
    to_drop = set(to_drop) & set(df.columns.tolist())
    text_data = df.drop(to_drop,axis=1)
    print(text_data.dtypes)
    # Replace nans with blanks
    text_data.fillna("",inplace=True)
    
    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)


# In[13]:


print(type(combine_text_columns(df)))


# In[14]:


from sklearn.preprocessing import FunctionTransformer


# In[15]:


# Preprocess the text data: get_text_data
get_text_data = FunctionTransformer(combine_text_columns, validate=False)

# Preprocess the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)


# In[16]:


# Fit and transform the text data: just_text_data
just_text_data = get_text_data.fit_transform(df)

# Fit and transform the numeric data: just_numeric_data
just_numeric_data = get_numeric_data.fit_transform(df)

# Print head to check results
print('Text Data')
print(just_text_data.head())
print('\nNumeric Data')
print(just_numeric_data.head())


# In[17]:


from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction.text import CountVectorizer


# In[18]:


#keep = list(NUMERIC_COLUMNS) + list(LABELS)
#df = df[keep]
#df.shape
#print(df.dtypes)


# In[19]:


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import make_scorer


# In[20]:


def multilabel_sample(y, size=1000, min_count=5, seed=None):
    """ Takes a matrix of binary labels `y` and returns
        the indices for a sample of size `size` if
        `size` > 1 or `size` * len(y) if size =< 1.
        The sample is guaranteed to have > `min_count` of
        each label.
    """
    try:
        if (np.unique(y).astype(int) != np.array([0, 1])).all():
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError('multilabel_sample only works with binary indicator matrices')

    if (y.sum(axis=0) < min_count).any():
        raise ValueError('Some classes do not have enough examples. Change min_count if necessary.')

    if size <= 1:
        size = np.floor(y.shape[0] * size)

    if y.shape[1] * min_count > size:
        msg = "Size less than number of columns * min_count, returning {} items instead of {}."
        warn(msg.format(y.shape[1] * min_count, size))
        size = y.shape[1] * min_count

    rng = np.random.RandomState(seed if seed is not None else np.random.randint(1))

    if isinstance(y, pd.DataFrame):
        choices = y.index
        y = y.values
    else:
        choices = np.arange(y.shape[0])

    sample_idxs = np.array([], dtype=choices.dtype)

    # first, guarantee > min_count of each label
    for j in range(y.shape[1]):
        label_choices = choices[y[:, j] == 1]
        label_idxs_sampled = rng.choice(label_choices, size=min_count, replace=False)
        sample_idxs = np.concatenate([label_idxs_sampled, sample_idxs])

    sample_idxs = np.unique(sample_idxs)

    # now that we have at least min_count of each, we can just random sample
    sample_count = int(size - sample_idxs.shape[0])

    # get sample_count indices from remaining choices
    remaining_choices = np.setdiff1d(choices, sample_idxs)
    remaining_sampled = rng.choice(remaining_choices,
                                   size=sample_count,
                                   replace=False)

    return np.concatenate([sample_idxs, remaining_sampled])


def multilabel_sample_dataframe(df, labels, size, min_count=5, seed=None):
    """ Takes a dataframe `df` and returns a sample of size `size` where all
        classes in the binary matrix `labels` are represented at
        least `min_count` times.
    """
    idxs = multilabel_sample(labels, size=size, min_count=min_count, seed=seed)
    return df.loc[idxs]


def multilabel_train_test_split(X, Y, size, min_count=5, seed=None):
    """ Takes a features matrix `X` and a label matrix `Y` and
        returns (X_train, X_test, Y_train, Y_test) where all
        classes in Y are represented at least `min_count` times.
    """
    index = Y.index if isinstance(Y, pd.DataFrame) else np.arange(Y.shape[0])

    test_set_idxs = multilabel_sample(Y, size=size, min_count=min_count, seed=seed)
    train_set_idxs = np.setdiff1d(index, test_set_idxs)

    test_set_mask = index.isin(test_set_idxs)
    train_set_mask = ~test_set_mask

    return (X[train_set_mask], X[test_set_mask], Y[train_set_mask], Y[test_set_mask])


# In[21]:


# Create training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(df[NON_LABELS],
                                                               df[LABELS],
                                                               size=0.2, 
                                                               seed=123)

# Print the info
print("X_train info:")
print(X_train.info())
print("\nX_test info:")  
print(X_test.info())
print("\ny_train info:")  
print(y_train.info())
print("\ny_test info:")  
print(y_test.info())


# In[22]:


print(type(X_train))


# In[24]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler

# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
pl = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ]))
                ,
                ('text_features', Pipeline([
                    ('selector', get_text_data),
#                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_ALPHANUMERIC, ngram_range=(1,2)))
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('scale', MaxAbsScaler()),
        ('clf', RandomForestClassifier(n_estimators = 400, max_depth = 4, 
                            n_jobs=-1, class_weight='balanced'))
    ])


# In[25]:


# Fit to the training data

pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on budget dataset: ", accuracy)


# In[26]:


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


# In[27]:


# get the predictions
def get_predictions(clf, X_test):
    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = clf.predict_proba(X_test)
    y_pred_prob = y_pred_prob[:,1]
    return y_pred_prob


# In[28]:


# print gini score
def print_scores(y_test, y_pred_prob):
    gini_predictions = gini(y_test, y_pred_prob)
    gini_max = gini(y_test, y_test)
    ngini= gini_sklearn(y_test, y_pred_prob)
    print('Normalized Gini: %.3f' 
      % (ngini))


# In[29]:


from sklearn.metrics import confusion_matrix,auc,roc_auc_score
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score


# In[30]:


def print_confusionmatrix(y_test,y_pred,y_pred_prob):
    print('test-set confusion matrix:\n', confusion_matrix(y_test,y_test)) 
    print('test-set confusion matrix:\n', confusion_matrix(y_test,y_pred)) 
    print("recall score: ", recall_score(y_test,y_pred))
    print("precision score: ", precision_score(y_test,y_pred))
    print("f1 score: ", f1_score(y_test,y_pred))
    print("accuracy score: ", accuracy_score(y_test,y_pred))
    print("ROC AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))


# In[31]:


y_pred = pl.predict(X_test)


# In[32]:


print(type(y_pred))


# In[33]:


y_pred_prob = pl.predict_proba(X_test)


# In[34]:


list_y_test = list(y_test.Class)
y_test.shape


# In[35]:


list_y_pred_prob = list(y_pred_prob[:,1])


# In[36]:


print_scores(list_y_test, list_y_pred_prob)


# In[37]:


print_confusionmatrix(list_y_test,y_pred,list_y_pred_prob)


# In[38]:


# Format predictions in DataFrame: prediction_df
prediction_df = pd.DataFrame(columns=df[LABELS].columns,
                             data=list_y_pred_prob)

prediction_df["known_label"] = list_y_test
# Save prediction_df to csv
prediction_df.to_csv('predictions.csv')


# Adding text features is not helping much, but still 5 out of 8 non-renewed cases are detected !
