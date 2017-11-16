
# coding: utf-8

# GOAL:
# 
# 1.      What insights can you draw from this data
# 
# 2.      What recommendation engine can you build from this data
# 
# 3.      What algorithms will make sense to come up with a churn model on whether the client will renew or not
# 
# 4.      Can you build segmentation or any propensity model with this data?
# 
# 5.      Anything else you can think of?

# In[1]:


import pandas as pd

filename = "/home/lovedeep/Desktop/HUB/lovedeep_data.xlsx"
df = pd.read_excel(filename)


# In[2]:


import matplotlib.pyplot as plt # plots


# I will first check the form and quality of data

# In[3]:


print("df shape {}".format(df.shape))


# this dataset has 10K instances, 28 features

# In[4]:


print("\ndf types")
print(df.dtypes)


# I have noticed that 9 of features are of numerical datatype ( float64 ) and 3 has dates ( datetime64 ), rest of them are encoded as "object". Let us check the data information about missing (non-null) entries.

# In[5]:


print("\ninfo")
df.info()


# We can see that there are few features with less than 10K entries, so we do have missing entries.
# 
# Let us look into head of dataset - first 5 lines, to see how the entries are written into columns.

# In[6]:


print("\nfirst 5 lines")
print(df.head(5))


# OK, so as noticed above, some of the features are numbers (float64), some are dates (datetime64), some are text (objects). Let us look at the statistical summary of these features, like mean/median/count/min/max etc. for the numerical features. For description of non-numeric columns, we will specify them separately after this. 

# In[7]:


print("\ndescriptive summary of numeric data")
print(df.describe())


# cool, no missing values in these numerical columns. But we can notice that range between min to max is quite large. Quite large range of outliers, need to play with data.
# 
# Note: three of features "AccrualAmt, BilledPremiumAmt, BilledCommissionAmt" have min as negative. Since this is about payments, so negative sounds strange.

# In[8]:


num_cat = df.dtypes[df.dtypes == 'float64'].index
print(num_cat)


# In[9]:


for col in num_cat:
    fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(10,3))
    df[col].plot(kind='density', ax = ax1)
    df[col].plot(kind='box', ax = ax2)
    ax3.hist(df[col], bins=100, alpha=0.5)
    plt.show()


# as noticed before, there are lots of outliers, so nothing can be seen in plots. Let us remove the outliers and plot again.

# In[10]:


def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


# In[11]:


for col in num_cat:
    if col == "DownloadedPremiumAmt":
        continue
    tmp_df = remove_outlier(df, col)
    fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(10,3))
    tmp_df[col].plot(kind='density', ax = ax1)
    tmp_df[col].plot(kind='box', ax = ax2)
    ax3.hist(tmp_df[col], bins=20, alpha=0.5)
    plt.show()


# In[12]:


import numpy as np
df_num = df[num_cat]
correlations = df_num.corr()
print(len(df_num.columns))
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
print(ticks)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(list(df_num.columns))
ax.set_yticklabels(list(df_num.columns))
plt.show()


# In[13]:


from pandas.plotting import scatter_matrix
scatter_matrix(df_num, alpha=0.2, figsize=(10, 10), diagonal='kde')
plt.show()


# Let us check count and unique values of "object" type categorical features. 

# In[14]:


print("\ndescriptive summary of non-numeric data")
obj_cat = df.dtypes[df.dtypes == 'object'].index
print(obj_cat)
print(df[obj_cat].describe())


# This gives me count of non-null records, and the number of unique categories, the most frequently occuring value, and the number of occurrences of most frequent value.
# 
# so, there are 16 categorical variables. Let us explore them one by one.
# 
# 1. ClientName
# "Michael Johnson" is there in 6 samples, it is just a chance or data replica? let us check.

# In[15]:


same_client = (df["ClientName"] == "Michael Johnson")
df[same_client]


# OK, different city, state, policy number etc., so this is not any mistake where someone copied the same data again and again. So it's fine. But since all the names are unique, and do not bring any predictive insight, we can drop it from dataset. On the other hand, we do have some information in names that we can use to distinuguish between sex (based on title or first name), ethnicity/race (based on family/last name) etc. We will check later.

# Next categorical features are city, state, postalcode etc. that can be helpful in record grouping or clustering etc. Note that there are some missing values too.  

# PolicyNumber that has more than 9K unique values, sounds like hard to interpret anything from it. We can drop it, but may be its combination with some other feature can provide some predictive hints. 
# 
# DepartmentName: one unique value "Commercial Lines" (Property/casualty insurance can be broken down into two major categories: commercial lines or types of insurance and personal lines. Personal lines, as the term suggests, includes coverages for individuals—auto and homeowners insurance. Commercial lines, which accounts for about half of U.S. property/casualty insurance industry premium, includes the many kinds of insurance products designed for businesses.)
# 
# DepartmentCode: one unique 'COM', so there is nothing informative in these two features, we can drop them from dataset easily.
# 
# CdPolicyLineTypeCode has 142 unique values with CPKG having frequency of 1835, this could be a useful categorical features. We will plot this interesting feature frequency table and plot later.
# 
# InsurerName has more than 700 unqiue values, but can be helpful to know the ones with larger market. A few of data instances has missing values for this feature.
# 
# BrokerName: A lot of missing values or may be data-instances donot have broker associated to them. 345 unique names, where AmWINS top group !
# 
# BillModes: 2 unique values: Direct or Agency bill. Direct has more than 6.6K instances.

# In[16]:


print(df.BillMode.unique())


# In[17]:


print(df.PolicyStatusName.nunique(), '\n', df.PolicyStatusName.unique())


# as noticed above, there are 12 different PolicyStatusNames. Renewal status with maximum frequency of 7596 instances. No null values. Interesting feature.

# In[18]:


print(df.CommissionTypeCode.unique())


# CommissionTypeCode with only 2 unique codes, no null values, interesting categorical !

# In[19]:


print(df.AgencyCode.nunique(), '\n', df.AgencyCode.unique())


# AgencyCode: 17 unique with 'CAL' having max frequency. no null values.

# In[20]:


print(df.BranchName.unique(), '\n', df.BranchName.nunique())


# BranchName has 218 unique names, 
# topper is Chicago, IL-Hub International Midwest West with 625 as max frequency.
# non-null values.

# In[21]:


print(df.ProducerName.unique(), '\n', df.ProducerName.nunique())


# 1008 unique names, non null. 
# topper Colorado House Account with 307 max frequency.

# In[22]:


print(df.isnull().sum())


# as noticed above, few of categorical variables has missing values. BrokerName may be is not meant tobe assigned to each insured, so has maximum missing values.

# In[23]:


print(df.PolicyStatusName.describe())
new_PolicyStatusName = pd.Categorical(df.PolicyStatusName)
print(new_PolicyStatusName.describe())


# In[24]:


from sklearn import preprocessing
le_PolicyStatusName = preprocessing.LabelEncoder()
le_PolicyStatusName.fit(df.PolicyStatusName)
le_PolicyStatusName.classes_


# In[25]:


df["LE_PolicyStatusName"] = le_PolicyStatusName.transform(df.PolicyStatusName)


# In[26]:


print(df.LE_PolicyStatusName.describe())


# In[27]:


useful_cat = ['State', 'BillMode', 'PolicyStatusName',
       'CommissionTypeCode', 'AgencyCode']
for cat in useful_cat:
    df[cat].value_counts().plot(kind="bar", figsize=(10,3))
    plt.show()

