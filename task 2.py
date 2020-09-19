#!/usr/bin/env python
# coding: utf-8

# # Linear Regression with Python Scikit Learn

# # Simple Linear Regression

# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# In[193]:


# import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[194]:


data_link="http://bit.ly/w-data"


# In[195]:


#read the data from url
data=pd.read_csv(data_link)
data.head()


# In[196]:


data.tail()


# In[197]:


data.describe()


# In[198]:


# Plotting the distribution of scores
plt.scatter(data["Hours"],data["Scores"])
plt.xlabel("hours")
plt.ylabel("Score")
plt.title("hours and score")
plt.grid()
plt.show()


# # Preparing the data

# In[199]:


x = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  


# In[200]:


x.shape


# In[201]:


y.shape


# In[202]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[203]:


x_train.shape


# In[204]:


y_train.shape


# In[205]:


x_test


# # Training  Algorithm

# In[206]:


#import linear regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
print("Training Completed")


# In[207]:


#predict score
y_predict=lr.predict(x_test)


# In[208]:


y_predict


# In[209]:


df=pd.DataFrame({"Actual":y_test,"Predicted":y_predict})


# In[210]:


df


# In[211]:


# You can also test with your own data
hours=9.25
own_pred=lr.predict([[hours]])
print(f"predict score for studying {hours} hr is :{own_pred}")


# In[212]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_predict)) 


# In[ ]:





# In[ ]:




