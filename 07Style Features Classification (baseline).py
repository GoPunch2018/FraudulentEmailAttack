#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
pd.options.display.max_columns = 250

import machine_learning as ml
from preprocessing import separate_features_target


# In[2]:


# Path
cwd = os.getcwd()
csv_path = os.path.join(cwd, 'data/csv/')

train = {
    'stylometric' : ['style_train_balanced.csv','style_train_imbalanced.csv']
}
test = {
    'stylometric' : ['style_test_balanced.csv','style_test_imbalanced.csv']
}


# # Balanced Dataset

# In[3]:


style_train_balanced_complete = pd.read_csv(os.path.join(csv_path, train['stylometric'][0]), index_col=0)
style_test_balanced_complete = pd.read_csv(os.path.join(csv_path, test['stylometric'][0]), index_col=0)


# In[4]:


style_train_balanced = separate_features_target(style_train_balanced_complete)
style_test_balanced = separate_features_target(style_test_balanced_complete)


# ## Train

# ### Logistic Regression

# In[5]:


lr_style_balanced = ml.train_logistic_regression(style_train_balanced['features'], style_train_balanced['target'], show_train_accuracy=1)
lr_style_balanced, lr_style_balanced_scaler = lr_style_balanced['model'], lr_style_balanced['scaler']


# ### Decision Tree

# In[6]:


dt_style_balanced = ml.train_decision_tree(style_train_balanced['features'], style_train_balanced['target'], show_train_accuracy=1)


# ### Random Forest

# In[7]:


rf_style_balanced = ml.train_random_forest(style_train_balanced['features'], style_train_balanced['target'], show_train_accuracy=1)


# ### Gradient Boosting Tree

# In[8]:


gb_style_balanced = ml.train_gradient_boost(style_train_balanced['features'], style_train_balanced['target'], show_train_accuracy=1)


# ### Naive Bayes

# In[9]:


nb_style_balanced = ml.train_naive_bayes(style_train_balanced['features'], style_train_balanced['target'], remove_negatives=True, show_train_accuracy=1)
nb_style_balanced, nb_style_balanced_scaler = nb_style_balanced['model'], nb_style_balanced['scaler']


# ## Test

# In[10]:


models = [lr_style_balanced, dt_style_balanced, rf_style_balanced, gb_style_balanced, nb_style_balanced]
names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting Tree', 'Naive Bayes']


# In[11]:


results_style_balanced = ml.multi_model_results(models, names, style_test_balanced['features'], style_test_balanced['target'], lr_style_balanced_scaler, nb_style_balanced_scaler)


# In[12]:


results_style_balanced


# # Imbalanced Dataset

# In[13]:


style_train_balanced_complete = pd.read_csv(os.path.join(csv_path, train['stylometric'][1]), index_col=0)
style_test_balanced_complete = pd.read_csv(os.path.join(csv_path, test['stylometric'][1]), index_col=0)


# In[14]:


style_train_imbalanced = separate_features_target(style_train_balanced_complete)
style_test_imbalanced = separate_features_target(style_test_balanced_complete)


# ## Train

# ### Logistic Regression

# In[15]:


lr_style_imbalanced = ml.train_logistic_regression(style_train_imbalanced['features'], style_train_imbalanced['target'], show_train_accuracy=1)
lr_style_imbalanced, lr_style_imbalanced_scaler = lr_style_imbalanced['model'], lr_style_imbalanced['scaler']


# ### Decision Tree

# In[16]:


dt_style_imbalanced = ml.train_decision_tree(style_train_imbalanced['features'], style_train_imbalanced['target'], show_train_accuracy=1)


# ### Random Forest

# In[17]:


rf_style_imbalanced = ml.train_random_forest(style_train_imbalanced['features'], style_train_imbalanced['target'], show_train_accuracy=1)


# ### Gradient Boosting Tree

# In[18]:


gb_style_imbalanced = ml.train_gradient_boost(style_train_imbalanced['features'], style_train_imbalanced['target'], show_train_accuracy=1)


# ### Naive Bayes

# In[19]:


nb_style_imbalanced = ml.train_naive_bayes(style_train_imbalanced['features'], style_train_imbalanced['target'], remove_negatives=True, show_train_accuracy=1)
nb_style_imbalanced, nb_style_imbalanced_scaler = nb_style_imbalanced['model'], nb_style_imbalanced['scaler']


# ## Test

# In[20]:


models = [lr_style_imbalanced, dt_style_imbalanced, rf_style_imbalanced, gb_style_imbalanced, nb_style_imbalanced]
names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting Tree', 'Naive Bayes']


# In[21]:


results_style_imbalanced = ml.multi_model_results(models, names, style_test_imbalanced['features'], style_test_imbalanced['target'], lr_style_imbalanced_scaler, nb_style_imbalanced_scaler)


# In[22]:


results_style_imbalanced


# # Comparison

# - This feature set is consistently a lot worse than the text content features. It seems like there may not be many strong predictors of phishing emails in the style of the author (which makes sense since authors of phishing emails try to imitate legitimate sources as much as possible), and the number of features was not big enough to make up for this.
# - However, the prediction capability is *still* a lot better than random chance (especially for the balanced dataset), so the combination of style and content featues might give better results.
# - The results were a lot worse in the imbalanced dataset.
# - Gradient Boosting was by far the best performing algorithm.
