#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
pd.options.display.max_columns = 250

import machine_learning as ml
from preprocessing import separate_features_target

from ast import literal_eval


# In[2]:


# Path
cwd = os.getcwd()
csv_path = os.path.join(cwd, 'data/csv/')

train = {
    'tfidf_sel' : ['tfidf_chi2_train_balanced.csv','tfidf_chi2_train_imbalanced.csv'],
    'word2vec' : ['word2vec_train_balanced.csv','word2vec_train_imbalanced.csv']
}
test = {
    'tfidf_sel' : ['tfidf_chi2_test_balanced.csv','tfidf_chi2_test_imbalanced.csv'],
    'word2vec' : ['word2vec_test_balanced.csv','word2vec_test_imbalanced.csv']
}


# # Balanced Dataset

# ## Train TF-IDF

# In[3]:


tfidf_train_balanced_complete = pd.read_csv(os.path.join(csv_path, train['tfidf_sel'][0]), index_col=0)
tfidf_test_balanced_complete = pd.read_csv(os.path.join(csv_path, test['tfidf_sel'][0]), index_col=0)


# In[4]:


tfidf_train_balanced = separate_features_target(tfidf_train_balanced_complete)
tfidf_test_balanced = separate_features_target(tfidf_test_balanced_complete)


# ### Logistic Regression

# In[5]:


lr_tfidf_balanced = ml.train_logistic_regression(tfidf_train_balanced['features'], tfidf_train_balanced['target'], show_train_accuracy=1)
lr_tfidf_balanced, lr_tfidf_balanced_scaler = lr_tfidf_balanced['model'], lr_tfidf_balanced['scaler']


# ### Decision Tree

# In[6]:


dt_tfidf_balanced = ml.train_decision_tree(tfidf_train_balanced['features'], tfidf_train_balanced['target'], show_train_accuracy=1)


# ### Random Forest

# In[7]:


rf_tfidf_balanced = ml.train_random_forest(tfidf_train_balanced['features'], tfidf_train_balanced['target'], show_train_accuracy=1)


# ### Gradient Boosting Tree

# In[8]:


gb_tfidf_balanced = ml.train_gradient_boost(tfidf_train_balanced['features'], tfidf_train_balanced['target'], show_train_accuracy=1)


# ### Naive Bayes

# In[9]:


nb_tfidf_balanced = ml.train_naive_bayes(tfidf_train_balanced['features'], tfidf_train_balanced['target'], show_train_accuracy=1)
nb_tfidf_balanced, nb_tfidf_balanced_scaler = nb_tfidf_balanced['model'], nb_tfidf_balanced['scaler']


# ## Train Word2Vec

# In[10]:


word2vec_train_balanced_complete = pd.read_csv(os.path.join(csv_path, train['word2vec'][0]), index_col=0)
word2vec_test_balanced_complete = pd.read_csv(os.path.join(csv_path, test['word2vec'][0]), index_col=0)


# In[11]:


word2vec_train_balanced = separate_features_target(word2vec_train_balanced_complete)
word2vec_test_balanced = separate_features_target(word2vec_test_balanced_complete)


# ### Logistic Regression

# In[12]:


lr_word2vec_balanced = ml.train_logistic_regression(word2vec_train_balanced['features'], word2vec_train_balanced['target'], show_train_accuracy=1)
lr_word2vec_balanced, lr_word2vec_balanced_scaler = lr_word2vec_balanced['model'], lr_word2vec_balanced['scaler']


# ### Decision Tree

# In[13]:


dt_word2vec_balanced = ml.train_decision_tree(word2vec_train_balanced['features'], word2vec_train_balanced['target'], show_train_accuracy=1)


# ### Random Forest

# In[14]:


rf_word2vec_balanced = ml.train_random_forest(word2vec_train_balanced['features'], word2vec_train_balanced['target'], show_train_accuracy=1)


# ### Gradient Boosting Tree

# In[15]:


gb_word2vec_balanced = ml.train_gradient_boost(word2vec_train_balanced['features'], word2vec_train_balanced['target'], show_train_accuracy=1)


# ### Naive Bayes

# In[16]:


nb_word2vec_balanced = ml.train_naive_bayes(word2vec_train_balanced['features'], word2vec_train_balanced['target'], remove_negatives=True, show_train_accuracy=1)
nb_word2vec_balanced, nb_word2vec_balanced_scaler = nb_word2vec_balanced['model'], nb_word2vec_balanced['scaler']


# ## Results

# ### TF-IDF

# In[17]:


models = [lr_tfidf_balanced, dt_tfidf_balanced, rf_tfidf_balanced, gb_tfidf_balanced, nb_tfidf_balanced]
names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting Tree', 'Naive Bayes']


# In[18]:


results_tfidf = ml.multi_model_results(models, names, tfidf_test_balanced['features'], tfidf_test_balanced['target'], lr_tfidf_balanced_scaler, nb_tfidf_balanced_scaler)


# In[19]:


results_tfidf


# ### Word2Vec

# In[20]:


models = [lr_word2vec_balanced, dt_word2vec_balanced, rf_word2vec_balanced, gb_word2vec_balanced, nb_word2vec_balanced]
names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting Tree', 'Naive Bayes']


# In[21]:


results_word2vec = ml.multi_model_results(models, names, word2vec_test_balanced['features'], word2vec_test_balanced['target'], lr_word2vec_balanced_scaler, nb_word2vec_balanced_scaler)


# In[22]:


results_word2vec


# # Imbalanced Dataset

# ## Train TF-IDF

# In[23]:


tfidf_train_imbalanced_complete = pd.read_csv(os.path.join(csv_path, train['tfidf_sel'][1]), index_col=0)
tfidf_test_imbalanced_complete = pd.read_csv(os.path.join(csv_path, test['tfidf_sel'][1]), index_col=0)


# In[24]:


tfidf_train_imbalanced = separate_features_target(tfidf_train_imbalanced_complete)
tfidf_test_imbalanced = separate_features_target(tfidf_test_imbalanced_complete)


# ### Logistic Regression

# In[25]:


lr_tfidf_imbalanced = ml.train_logistic_regression(tfidf_train_imbalanced['features'], tfidf_train_imbalanced['target'], show_train_accuracy=1)
lr_tfidf_imbalanced, lr_tfidf_imbalanced_scaler= lr_tfidf_imbalanced['model'], lr_tfidf_imbalanced['scaler']


# ### Decision Tree

# In[26]:


dt_tfidf_imbalanced = ml.train_decision_tree(tfidf_train_imbalanced['features'], tfidf_train_imbalanced['target'], show_train_accuracy=1)


# ### Random Forest

# In[27]:


rf_tfidf_imbalanced = ml.train_random_forest(tfidf_train_imbalanced['features'], tfidf_train_imbalanced['target'], show_train_accuracy=1)


# ### Gradient Boosting Tree

# In[28]:


gb_tfidf_imbalanced = ml.train_gradient_boost(tfidf_train_imbalanced['features'], tfidf_train_imbalanced['target'], show_train_accuracy=1)


# ### Naive Bayes

# In[29]:


nb_tfidf_imbalanced = ml.train_naive_bayes(tfidf_train_imbalanced['features'], tfidf_train_imbalanced['target'], show_train_accuracy=1)
nb_tfidf_imbalanced, nb_tfidf_imbalanced_scaler = nb_tfidf_imbalanced['model'], nb_tfidf_imbalanced['scaler']


# ## Train Word2Vec

# In[30]:


word2vec_train_imbalanced_complete = pd.read_csv(os.path.join(csv_path, train['word2vec'][1]), index_col=0)
word2vec_test_imbalanced_complete = pd.read_csv(os.path.join(csv_path, test['word2vec'][1]), index_col=0)


# In[31]:


word2vec_train_imbalanced = separate_features_target(word2vec_train_imbalanced_complete)
word2vec_test_imbalanced = separate_features_target(word2vec_test_imbalanced_complete)


# ### Logistic Regression

# In[32]:


lr_word2vec_imbalanced = ml.train_logistic_regression(word2vec_train_imbalanced['features'], word2vec_train_imbalanced['target'], show_train_accuracy=1)
lr_word2vec_imbalanced, lr_word2vec_imbalanced_scaler= lr_word2vec_imbalanced['model'], lr_word2vec_imbalanced['scaler']


# ### Decision Tree

# In[33]:


dt_word2vec_imbalanced = ml.train_decision_tree(word2vec_train_imbalanced['features'], word2vec_train_imbalanced['target'], show_train_accuracy=1)


# ### Random Forest

# In[34]:


rf_word2vec_imbalanced = ml.train_random_forest(word2vec_train_imbalanced['features'], word2vec_train_imbalanced['target'], show_train_accuracy=1)


# ### Gradient Boosting Tree

# In[35]:


gb_word2vec_imbalanced = ml.train_gradient_boost(word2vec_train_imbalanced['features'], word2vec_train_imbalanced['target'], show_train_accuracy=1)


# ### Naive Bayes

# In[36]:


nb_word2vec_imbalanced = ml.train_naive_bayes(word2vec_train_imbalanced['features'], word2vec_train_imbalanced['target'], remove_negatives=True, show_train_accuracy=1)
nb_word2vec_imbalanced, nb_word2vec_imbalanced_scaler = nb_word2vec_imbalanced['model'], nb_word2vec_imbalanced['scaler']


# ## Results

# ### TF-IDF

# In[37]:


models = [lr_tfidf_imbalanced, dt_tfidf_imbalanced, rf_tfidf_imbalanced, gb_tfidf_imbalanced, nb_tfidf_imbalanced]
names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting Tree', 'Naive Bayes']


# In[38]:


results_tfidf_imbalanced = ml.multi_model_results(models, names, tfidf_test_imbalanced['features'], tfidf_test_imbalanced['target'], lr_tfidf_imbalanced_scaler, nb_tfidf_imbalanced_scaler)


# In[39]:


results_tfidf_imbalanced


# ### Word2Vec

# In[40]:


models = [lr_word2vec_imbalanced, dt_word2vec_imbalanced, rf_word2vec_imbalanced, gb_word2vec_imbalanced, nb_word2vec_imbalanced]
names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting Tree', 'Naive Bayes']


# In[41]:


results_word2vec_imbalanced = ml.multi_model_results(models, names, word2vec_test_imbalanced['features'], word2vec_test_imbalanced['target'], lr_word2vec_imbalanced_scaler, nb_word2vec_imbalanced_scaler)


# In[42]:


results_word2vec_imbalanced


# # Specific emails

# It is possible to see the predictions of each algorithm with Word2Vec features (since those performed the best) for selected emails (some of which were also seen previously).

# In[43]:


models = [lr_word2vec_balanced, dt_word2vec_balanced, rf_word2vec_balanced, gb_word2vec_balanced, nb_word2vec_balanced]
names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting Tree', 'Naive Bayes']


# In[44]:


ml.results_by_id(models, names, word2vec_test_balanced_complete, [5, 17, 1379], lr_word2vec_balanced_scaler, nb_word2vec_balanced_scaler)


# We see that the emails with `id: 17` was correctly identified as phishing by all algorithms and, similarly, the email with `id: 5` was unanimously correctly identified as legitimate.<br>
# 
# On the other hand, the email with `id: 1379` was misclassified by all of the algorithms except Logistic Regression.
# ```
# Hello,
#  I hope you are safe from the Covid 19.
#  We are currently back to work and our company hope to place our urgent orders as previously discussed before the lockdown.
# Kindly find below our attached order via Wetransfer and confirm availability of all products.
# 
# 
#  https://wetransfer.com/downloads
# 
# 
# 
#  Kindly send in your best quote and shortest delivery time.
#  -
# 
# 
#  Greetings!
# Maria Pietrygas (Import Manager)
# Nautril Holdings
# Athens Gr
# Tel.08872917845
# ```
# Indeed, while it is a phishing email, it appears to be legitimate (despite the fact that this company does not even exist). Apart from a sense of urgency, none of the other usual phishing markers are present.

# # Conclusions

# - As expected, the algorithms performed better on the balanced dataset.
# - The best performing algorithms were Gradient Boosting and Logistic Regression. Gradient Boosting was a bit more consistent and achieved the best results in the balanced dataset, but Logistic Regression outperformed it in on the imbalanced set.
# - Naive Bayes (despite achieving the best results in balanced TF-IDF) is not very well suited to such classification problems, and is especially bad with imbalanced datasets.
# - Wor2Vec features definitely outperformed TF-IDF, both in balanced and imbalanced datasets.
