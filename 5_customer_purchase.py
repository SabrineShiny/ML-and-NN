#!/usr/bin/env python
# coding: utf-8

# ### 五、
# 关于潜在顾客的购买可能的估计问题。每个顾客的信息包括：年龄（Age）、收入情况(Incoming)、是否是学生(Is Student?)、信用等级(Credit Rating)。是通过下表（表一）中的数据建立Decision Tree，并判断如下用户的购买可能。新用户的信息如下：年龄(50), 收入（Medium）,非学生，信用记录 (excellent).

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz


# In[2]:


age  = ["less30","less30","31_40","more40","more40","more40","31_40","less30","less30",
        "more40","less30","31_40","31_40"]
income = ["high","high","high","medium","low","low","low","medium","low","medium",
         "medium","medium","high"]
isstudent = ["no","no","no","no","yes","yes","yes","no","yes","yes","yes","no","yes"]
credit = ["fair","excellent","fair","fair","fair","excellent","excellent","fair","fair","fair",
         "excellent","excellent","fair"]
buy = ["n","n","y","y","y","n","y","n","y","y","y","y","y"]
dic ={"age":age,"income":income,"isstudent":isstudent,"credit":credit,"buy":buy}


# In[3]:


data = pd.DataFrame()
for key in dic:
    encoder = LabelEncoder()
    encoder.fit(dic[key])
    data[key]= encoder.transform(dic[key])
data


# In[4]:


#构建决策树模型并训练
clf = DecisionTreeClassifier(random_state=1)
clf.fit(data.iloc[:,:-1], data["buy"])
print(clf.score(data.iloc[:,:-1], data["buy"]))


# In[5]:


#树的特征
print(clf.get_depth())
print(clf.get_n_leaves())
print(clf.tree_.node_count)


# In[6]:


#决策树模型可视化
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=list(dic.keys())[:-1],  
                     class_names=["no","yes"],  
                     filled=True, rounded=True)  
graph = graphviz.Source(dot_data)
graph


# In[7]:


#预测
new_data = np.array([[2,2,0,0]])
clf.predict(new_data)


# 由决策树的结果，新用户不会购买
