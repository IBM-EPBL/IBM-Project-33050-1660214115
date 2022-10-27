#!/usr/bin/env python
# coding: utf-8

# # 1. Load the dataset into the tool

# In[1]:


#importing the libraries

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


#loading the dataset

d = pd.read_csv(r'C:/Users/smiwin/OneDrive/Desktop/IBM Datasets/abalone.csv')


# # 2. Perform Below Visualizations

# ## ∙Univariate Analysis
# 

# In[3]:


d.head()


# In[4]:


#Boxplot

sns.boxplot(d['Diameter'])


# In[5]:


#histogram

plt.hist(d['Diameter'])


# In[6]:


#line plot

plt.plot(d['Diameter'].head(10))


# In[7]:


#piechart

plt.pie(d['Diameter'].head(),autopct='%.2f')


# In[8]:


#distplot

sns.distplot(d['Diameter'].head(200))


# ## ∙ Bi-Variate Analysis

# In[9]:


#scatter plot

plt.scatter(d['Diameter'].head(500),d['Length'].head(500))


# In[10]:


#bar plot

plt.bar(d['Sex'].head(10),d['Rings'].head(10))

#labelling of x,y and result

plt.title('Bar plot')
plt.xlabel('Diameter')
plt.ylabel('Rings')


# In[11]:


sns.barplot(d['Sex'], d['Rings'])


# In[12]:


#joint plot

sns.jointplot(d['Diameter'].head(50),d['Rings'].head(50))


# In[13]:


#bar plot

sns.barplot('Diameter','Rings',hue='Sex',data=d.head())


# In[14]:


sns.lineplot(d['Diameter'].head(),d['Rings'].head())


# ## ∙ Multi-Variate Analysis

# In[15]:


#boxplot

sns.boxplot(d['Sex'].head(10),d['Diameter'].head(10),d['Rings'].head(10))


# In[16]:


#heat map

fig=plt.figure(figsize=(8,5))
sns.heatmap(d.head().corr(),annot=True)


# In[17]:


#pair plot

sns.pairplot(d.head(),hue='Rings')


# In[18]:


sns.pairplot(d.head())


# # 3. Perform descriptive statistics on the dataset.
# 

# In[19]:


d.head()


# In[20]:


d.tail()


# In[21]:


d.info()


# In[22]:


d.describe()


# In[23]:


d.mode().T


# In[24]:


d.shape


# In[25]:


d.kurt()


# In[26]:


d.skew()


# In[27]:


d.var()


# In[28]:


d.nunique()


# # 4. Check for Missing values and deal with them.
# 

# In[30]:


#finding missing values

d.isna()


# In[31]:


d.isna().any()


# In[32]:


d.isna().sum()


# In[33]:


d.isna().any().sum()
#no missing values


# # 5. Find the outliers and replace them outliers

# In[34]:


#finding outliers

sns.boxplot(d['Diameter'])


# In[35]:


#handling outliers

qnt=d.quantile(q=[0.25,0.75])
qnt


# In[36]:


iqr=qnt.loc[0.75]-qnt.loc[0.25]

iqr


# In[37]:


lower=qnt.loc[0.25]-(1.5*iqr)
lower


# In[38]:


upper=qnt.loc[0.75]+(1.5*iqr)
upper


# In[39]:


# replacing outliers

##Diameter
d['Diameter']=np.where(d['Diameter']<0.155,0.4078,d['Diameter'])
sns.boxplot(d['Diameter'])


# In[40]:


## Length

sns.boxplot(d['Length'])


# In[41]:


d['Length']=np.where(d['Length']<0.23,0.52, d['Length'])


# In[42]:


sns.boxplot(d['Length'])


# In[43]:


## Height

sns.boxplot(d['Height'])


# In[44]:


d['Height']=np.where(d['Height']<0.04,0.139, d['Height'])
d['Height']=np.where(d['Height']>0.23,0.139, d['Height'])


# In[45]:


sns.boxplot(d['Height'])


# In[46]:


## Whole weight

sns.boxplot(d['Whole weight'])


# In[47]:


d['Whole weight']=np.where(d['Whole weight']>0.9,0.82, d['Whole weight'])


# In[48]:


sns.boxplot(d['Whole weight'])


# In[49]:


## Shucked weight

sns.boxplot(d['Shucked weight'])


# In[50]:


d['Shucked weight']=np.where(d['Shucked weight']>0.93,0.35, d['Shucked weight'])


# In[51]:


sns.boxplot(d['Shucked weight'])


# In[52]:


## Viscera weight

sns.boxplot(d['Viscera weight'])


# In[53]:


d['Viscera weight']=np.where(d['Viscera weight']>0.46,0.18, d['Viscera weight'])


# In[54]:


sns.boxplot(d['Viscera weight'])


# In[55]:


## Shell weight

sns.boxplot(d['Shell weight'])


# In[56]:


d['Shell weight']=np.where(d['Shell weight']>0.61,0.2388, d['Shell weight'])


# In[57]:


sns.boxplot(d['Shell weight'])


# # 6. Check for Categorical columns and perform encoding.

# In[58]:


#one hot encoding

d['Sex'].replace({'M':1,'F':0,'I':2},inplace=True)
d


# # 7. Split the data into dependent and independent variables.

# In[59]:


x=d.drop(columns= ['Rings'])
y=d['Rings']
x


# In[60]:


y


# # 8. Scale the independent variables

# In[61]:


from sklearn.preprocessing import scale  #StandardScaler


# In[62]:


#Scaling the independent variables

x = scale(x)
x


# # 9. Split the data into training and testing
# 

# In[63]:


from sklearn.model_selection import train_test_split


# In[64]:


#spliting data to train and test

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)
print(x_train.shape, x_test.shape)


# # 10. Build the Model
# 

# In[66]:


#Multiple Regression 

from sklearn.linear_model import LinearRegression

MLR=LinearRegression()


# # 11. Train the model

# In[68]:


MLR.fit(x_train,y_train)


# # 12. Test the model

# In[69]:


#predcition on the test data
y_pred=MLR.predict(x_test)
y_pred


# In[70]:


#prediction in the train data 
pred=MLR.predict(x_train)
pred


# In[71]:


from sklearn.metrics import r2_score

acc=r2_score(y_test,y_pred)

acc


# In[72]:


#test this model

MLR.predict([[1,0.455,0.365,0.095,0.5140,0.2245,0.1010,0.150]])


# # 13. Measure the performance using Metrics

# In[73]:


from sklearn import metrics
from sklearn.metrics import mean_squared_error


# In[74]:


np.sqrt(mean_squared_error(y_test,y_pred))


# ## LASSO

# In[75]:


from sklearn.linear_model import Lasso, Ridge


# In[76]:


#intialising model

lso=Lasso(alpha=0.01,normalize=True)


# In[77]:


#fit the model
lso.fit(x_train,y_train)


# In[78]:


#predcition on test data

lso_pred=lso.predict(x_test)


# In[79]:


#coef
coef=lso.coef_
coef


# In[80]:


#accuracy

from sklearn import metrics
from sklearn.metrics import mean_squared_error
metrics.r2_score(y_test,lso_pred)


# In[81]:



#error

np.sqrt(mean_squared_error(y_test,lso_pred))


# ## RIDGE

# In[82]:


rg=Ridge(alpha=0.01,normalize=True)


# In[83]:


#fit
rg.fit(x_train,y_train)


# In[84]:


#predcition

rg_pred=rg.predict(x_test)
rg_pred


# In[85]:


#coef
rg.coef_


# In[86]:


#accuracy

metrics.r2_score(y_test,rg_pred)


# In[87]:


#error

np.sqrt(mean_squared_error(y_test,rg_pred))

