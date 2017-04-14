
# coding: utf-8

# In[2]:

from __future__ import division, print_function
# отключим всякие предупреждения Anaconda
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic(u'pylab inline')

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

import seaborn as sns
import math
import pandas as pd
import networkx as nx
import itertools as IT
import numpy as np
import random as rnd 
import pylab as plt
get_ipython().magic(u'pylab inline')
import re


# In[97]:

graph = nx.read_edgelist("C:/Users/MsSurgeon/Documents/BeelineHW/Data/social_network_11.csv",delimiter=',')
edge = pd.read_csv("C:/Users/MsSurgeon/Documents/BeelineHW/Data/social_network_11.csv", names=['i','j'], dtype='str')
edge = edge[edge['i'] != edge['j']]
edge.index = np.arange(0, len(edge))
edge['y']=1


# In[71]:

counter = 0
missing = []
for pair in IT.combinations(graph.nodes(), 2):
    counter = counter + 1
    if counter > 1000000:
        break
    if not graph.has_edge(*pair):
        missing.append(pair)


# In[72]:

missing = rnd.sample(missing,35000)
missing = pd.DataFrame(missing,columns=['i','j'], dtype='str')
missing['y'] = 0


# In[73]:

df = pd.concat([edge, missing], ignore_index=True)


# In[74]:

df = df.sample(frac=1).reset_index(drop=True)


# In[75]:

removeEdge = edge.sample(int(round(len(edge)*0.2)))


# In[76]:

len(edge)


# In[77]:

len(removeEdge)


# In[78]:

graph.number_of_edges()


# In[95]:

graph.has_edge('2414','2415')


# In[101]:

for index, rE in removeEdge.iterrows():
    graph.remove_edge(str(rE['i']), str(rE['j']))


# In[72]:

df['i_Degree'] = df['i'].apply(lambda x: graph.degree(x))
df['j_Degree'] = df['j'].apply(lambda x: graph.degree(x))


# In[73]:

def GraphDist(row):
    try:
        return  nx.shortest_path_length(graph,row['i'],row['j'])
    except nx.NetworkXNoPath:
        return -1
        
df['GraphDist'] = df.apply(GraphDist, axis=1)


# In[74]:

def CommonN(row):
    try:
        return  len(sorted(nx.common_neighbors(graph,row['i'],row['j'])))
    except nx.NetworkXNoPath:
        return -1
df['CommonN'] = df.apply(CommonN, axis=1)


# In[75]:

df['Jaccard'] = df['CommonN']/(df['i_Degree']+df['i_Degree']-df['CommonN'])


# In[76]:

df['Sorensen'] = 2*df['CommonN']/(df['i_Degree']+df['i_Degree'])


# In[77]:

df['LHNI'] = df['CommonN']/(df['i_Degree']*df['i_Degree'])


# In[78]:

df['HPI'] = df['CommonN']/min(df['i_Degree']*df['i_Degree'])


# In[79]:

df['HDI'] = df['CommonN']/max(df['i_Degree']*df['i_Degree'])


# In[80]:

def Adamic_Adar(row):
    summ = 0
    try:
        neighbors = nx.common_neighbors(graph,row['i'],row['j'])
    except nx.NetworkXNoPath:
        return -1
    for n in neighbors:
        try:
            A_A = 1/math.log(graph.degree(n))
        except nx.NetworkXNoPath:
            return -1
        summ = summ + A_A
    return summ
df['Adamic_Adar'] = df.apply(Adamic_Adar, axis=1)


# In[81]:

def ResourceAllocation(row):
    summ = 0
    try:
        neighbors = nx.common_neighbors(graph,row['i'],row['j'])
    except nx.NetworkXNoPath:
        return -1
    for n in neighbors:
        try:
            A_A = 1/abs(graph.degree(n))
        except nx.NetworkXNoPath:
            return -1
        summ = summ + A_A
    return summ
df['ResourceAllocation'] = df.apply(ResourceAllocation, axis=1)


# In[82]:

df['PrefAttach'] = df['i_Degree']*df['i_Degree']


# In[83]:

df.head(20).T


# In[84]:

y = df['y']
df = df.drop(['y','i','j'], axis=1)


# In[85]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV

from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


# In[86]:

def plot_with_err(x, data, **kwargs):
    mu, std = data.mean(1), data.std(1)
    lines = plt.plot(x, mu, '-', **kwargs)
    plt.fill_between(x, mu - std, mu + std, edgecolor='none',
                     facecolor=lines[0].get_color(), alpha=0.2)
#////////////////////////////////////////////////////////////////////////////////////////////////    
train_sizes = np.linspace(0.05, 1, 20)
N_train, val_train, val_test = learning_curve(RandomForestClassifier(max_depth=7,                                                    min_samples_leaf=8,                                                    n_estimators = 10,                                                    random_state=555),                                                    df, y, train_sizes, cv=5,                                                    scoring='roc_auc')
#////////////////////////////////////////////////////////////////////////////////////////////////    
plt.figure(figsize=(15, 6))
plot_with_err(N_train, val_train, label='training scores')
plot_with_err(N_train, val_test, label='validation scores')
plt.xlabel('Training Set Size'); plt.ylabel('ROC AUC')
plt.legend()


# ### Обнаружилось что с метрикой GraphDist модель переобучаеться

# In[87]:

df = df.drop(['GraphDist'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=42)


# In[88]:

def plot_with_err(x, data, **kwargs):
    mu, std = data.mean(1), data.std(1)
    lines = plt.plot(x, mu, '-', **kwargs)
    plt.fill_between(x, mu - std, mu + std, edgecolor='none',
                     facecolor=lines[0].get_color(), alpha=0.2)
#////////////////////////////////////////////////////////////////////////////////////////////////    
train_sizes = np.linspace(0.05, 1, 20)
N_train, val_train, val_test = learning_curve(RandomForestClassifier(max_depth=7,                                                    min_samples_leaf=8,                                                    n_estimators = 10,                                                    random_state=555),                                                    df, y, train_sizes, cv=5,                                                    scoring='roc_auc')
#////////////////////////////////////////////////////////////////////////////////////////////////    
plt.figure(figsize=(15, 6))
plot_with_err(N_train, val_train, label='training scores')
plot_with_err(N_train, val_test, label='validation scores')
plt.xlabel('Training Set Size'); plt.ylabel('ROC AUC')
plt.legend()


# In[89]:

clf_params = [{'max_depth': list(range(5,10)), 'min_samples_leaf': list(range(5,10))}]
#forest = RandomizedSearchCV(RandomForestClassifier(n_estimators = 10,random_state=555),clf_params,\
#                                                      scoring="f1",cv=5)

#forest = GridSearchCV(RandomForestClassifier(n_estimators = 10,random_state=555),clf_params,\
#                                                      scoring="f1",cv=5)

forest = RandomForestClassifier(n_estimators = 10,random_state=555, max_depth=7, min_samples_leaf=8)
forest.fit(X_train, y_train)
#////////////////////////////////////////////////////////////////////////////////////////////////
y_pred = forest.predict(X_test)
y_probs = forest.predict_proba(X_test)
Acc = accuracy_score(y_test, y_pred)
ROC = roc_auc_score(y_test, y_probs[:, 1])
f1 = f1_score(y_test, y_pred)

#scores.append(clf_score)
print (' Acuuracy: ' + str(Acc) + ' ROC: ' + str(ROC) + ' f1: ' + str(f1))
#////////////////////////////////////////////////////////////////////////////////////////////////
print (pd.DataFrame(forest.feature_importances_, columns=['Importance'],             index=df.columns).sort_values(by='Importance', ascending=False))
#////////////////////////////////////////////////////////////////////////////////////////////////    


# In[61]:

r = pd.DataFrame(y_test + y_pred)
r.groupby('y').size()


# In[ ]:



