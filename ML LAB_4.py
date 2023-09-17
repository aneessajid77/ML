#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import mutual_info_classif

 

# Sample data table (you should replace this with your own data)
data = pd.DataFrame({
    'age': ['<=30', '<=30', '31-40', '>40', '>40', '>40', '31-40', '<=30', '<=30', '>40', '<=30', '31-40', '31-40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
})

 

# Convert categorical data to numerical using one-hot encoding
data = pd.get_dummies(data, drop_first=True)

 

# Split data into features and target variable
X = data.drop('buys_computer_yes', axis=1)
y = data['buys_computer_yes']

 

# Calculate Information Gain for each feature
information_gain = mutual_info_classif(X, y)

 

# Find the feature with the highest Information Gain
root_node_feature_index = np.argmax(information_gain)
root_node_feature_name = X.columns[root_node_feature_index]

 

print(f"The root node feature selected is: {root_node_feature_name}")


# In[ ]:




