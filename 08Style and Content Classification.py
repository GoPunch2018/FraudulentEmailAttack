#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
pd.options.display.max_columns = 250

import machine_learning as ml
from preprocessing import separate_features_target

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


# In[ ]:





# In[2]:


# Path
cwd = os.getcwd()
csv_path = os.path.join(cwd, 'data/csv/')

train = {
    'stylometric' : ['style_train_balanced.csv','style_train_imbalanced.csv'],
    'word2vec' : ['word2vec_train_balanced.csv','word2vec_train_imbalanced.csv']
}
test = {
    'stylometric' : ['style_test_balanced.csv','style_test_imbalanced.csv'],
    'word2vec' : ['word2vec_test_balanced.csv','word2vec_test_imbalanced.csv']
}


# ## Import Data

# Since Word2Vec features outperformed the TF-IDF features, only those will be used to test the combination with content features.

# ### Balanced Dataset

# In[3]:


style_train_balanced_complete = pd.read_csv(os.path.join(csv_path, train['stylometric'][0]), index_col=0, dtype={'email_class': 'bool', 'email_id': 'int16'})
style_test_balanced_complete = pd.read_csv(os.path.join(csv_path, test['stylometric'][0]), index_col=0, dtype={'email_class': 'bool', 'email_id': 'int16'})

word2vec_train_balanced_complete = pd.read_csv(os.path.join(csv_path, train['word2vec'][0]), index_col=0, dtype={'email_class': 'bool', 'email_id': 'int16'})
word2vec_test_balanced_complete = pd.read_csv(os.path.join(csv_path, test['word2vec'][0]), index_col=0, dtype={'email_class': 'bool', 'email_id': 'int16'})


# In[4]:


style_train_balanced = separate_features_target(style_train_balanced_complete)
style_test_balanced = separate_features_target(style_test_balanced_complete)

word2vec_train_balanced = separate_features_target(word2vec_train_balanced_complete)
word2vec_test_balanced = separate_features_target(word2vec_test_balanced_complete)


# ### Imbalanced Dataset

# In[5]:


style_train_imbalanced_complete = pd.read_csv(os.path.join(csv_path, train['stylometric'][1]), index_col=0, dtype={'email_class': 'bool', 'email_id': 'int16'})
style_test_imbalanced_complete = pd.read_csv(os.path.join(csv_path, test['stylometric'][1]), index_col=0, dtype={'email_class': 'bool', 'email_id': 'int16'})

word2vec_train_imbalanced_complete = pd.read_csv(os.path.join(csv_path, train['word2vec'][1]), index_col=0, dtype={'email_class': 'bool', 'email_id': 'int16'})
word2vec_test_imbalanced_complete = pd.read_csv(os.path.join(csv_path, test['word2vec'][1]), index_col=0, dtype={'email_class': 'bool', 'email_id': 'int16'})


# In[6]:


style_train_imbalanced = separate_features_target(style_train_imbalanced_complete)
style_test_imbalanced = separate_features_target(style_test_imbalanced_complete)

word2vec_train_imbalanced = separate_features_target(word2vec_train_imbalanced_complete)
word2vec_test_imbalanced = separate_features_target(word2vec_test_imbalanced_complete)


# # Merging Feature Sets

# The simplest way of combining the information of the two different feature sets is to simply merge them into one set and then perform the predictions based on this concatenated set.

# ## Balanced Dataset

# In[7]:


style_content_train_balanced = pd.concat([word2vec_train_balanced['features'], style_train_balanced['features']], axis=1)
style_content_test_balanced = pd.concat([word2vec_test_balanced['features'], style_test_balanced['features']], axis=1)


# ### Train

# #### Logistic Regression

# In[8]:


get_ipython().run_cell_magic('time', '', "lr_style_content_balanced = ml.train_logistic_regression(style_content_train_balanced, style_train_balanced['target'], show_train_accuracy=1)\nlr_style_content_balanced, lr_style_content_balanced_scaler = lr_style_content_balanced['model'], lr_style_content_balanced['scaler']\n")


# #### Decision Tree

# In[9]:


get_ipython().run_cell_magic('time', '', "dt_style_content_balanced = ml.train_decision_tree(style_content_train_balanced, style_train_balanced['target'], show_train_accuracy=1)\n")


# #### Random Forest

# In[10]:


get_ipython().run_cell_magic('time', '', "rf_style_content_balanced = ml.train_random_forest(style_content_train_balanced, style_train_balanced['target'], show_train_accuracy=1)\n")


# #### Gradient Boosting

# In[11]:


get_ipython().run_cell_magic('time', '', "gb_style_content_balanced = ml.train_gradient_boost(style_content_train_balanced, style_train_balanced['target'], show_train_accuracy=1)\n")


# #### Naive Bayes

# In[12]:


get_ipython().run_cell_magic('time', '', "nb_style_content_balanced = ml.train_naive_bayes(style_content_train_balanced, style_train_balanced['target'], show_train_accuracy=1, remove_negatives=True)\nnb_style_content_balanced, nb_style_content_balanced_scaler = nb_style_content_balanced['model'], nb_style_content_balanced['scaler']\n")


# ### Results

# In[13]:


models = [lr_style_content_balanced, dt_style_content_balanced, rf_style_content_balanced, gb_style_content_balanced, nb_style_content_balanced]
names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting Tree', 'Naive Bayes']


# In[14]:


get_ipython().run_cell_magic('time', '', "results_style_content_balanced = ml.multi_model_results(models, names, style_content_test_balanced, style_test_balanced['target'], lr_style_content_balanced_scaler, nb_style_content_balanced_scaler)\n")


# In[15]:


results_style_content_balanced


# ## Imbalanced Dataset

# In[16]:


style_content_train_imbalanced = pd.concat([word2vec_train_imbalanced['features'], style_train_imbalanced['features']], axis=1)
style_content_test_imbalanced = pd.concat([word2vec_test_imbalanced['features'], style_test_imbalanced['features']], axis=1)


# ### Train

# #### Logistic Regression

# In[17]:


get_ipython().run_cell_magic('time', '', "lr_style_content_imbalanced = ml.train_logistic_regression(style_content_train_imbalanced, style_train_imbalanced['target'], show_train_accuracy=1)\nlr_style_content_imbalanced, lr_style_content_imbalanced_scaler = lr_style_content_imbalanced['model'], lr_style_content_imbalanced['scaler']\n")


# #### Decision Tree

# In[18]:


get_ipython().run_cell_magic('time', '', "dt_style_content_imbalanced = ml.train_decision_tree(style_content_train_imbalanced, style_train_imbalanced['target'], show_train_accuracy=1)\n")


# #### Random Forest

# In[19]:


get_ipython().run_cell_magic('time', '', "rf_style_content_imbalanced = ml.train_random_forest(style_content_train_imbalanced, style_train_imbalanced['target'], show_train_accuracy=1)\n")


# #### Gradient Boosting

# In[20]:


get_ipython().run_cell_magic('time', '', "gb_style_content_imbalanced = ml.train_gradient_boost(style_content_train_imbalanced, style_train_imbalanced['target'], show_train_accuracy=1)\n")


# #### Naive Bayes

# In[21]:


get_ipython().run_cell_magic('time', '', "nb_style_content_imbalanced = ml.train_naive_bayes(style_content_train_imbalanced, style_train_imbalanced['target'], show_train_accuracy=1, remove_negatives=True)\nnb_style_content_imbalanced, nb_style_content_imbalanced_scaler = nb_style_content_imbalanced['model'], nb_style_content_imbalanced['scaler']\n")


# ### Results

# In[22]:


models = [lr_style_content_imbalanced, dt_style_content_imbalanced, rf_style_content_imbalanced, gb_style_content_imbalanced, nb_style_content_imbalanced]
names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting Tree', 'Naive Bayes']


# In[23]:


get_ipython().run_cell_magic('time', '', "results_style_content_imbalanced = ml.multi_model_results(models, names, style_content_test_imbalanced, style_test_imbalanced['target'], lr_style_content_imbalanced_scaler, nb_style_content_imbalanced_scaler)\n")


# In[24]:


results_style_content_imbalanced


# Comparing these with the content-only baseline, it is obvious that there is a at small improvement with GB and RF on the balanced dataset, while all algorithms except RF improved on the imbalanced dataset.<br>
# It seems that the extra features are helpful with the more robust algoritms in order to achieve better accuracy on the bigger dataset.

# # Stacking

# In machine learning, stacking refers to the proccess of using different learners (each one working best at learning a different part of the problem) called level 0 models as intermediate steps and then use their outputs to train another learner, called level 1 model. Thus, the final model is sometimes able to outperform the individual ones.

# On this specific case, the different initial classifiers will be trained on both of the feature sets, and thus the final classifier essentially will combine information from both of them.

# #### Final Classifiers

# Only the three best classifiers will be used as a level 1 classifier, namely Logistic Regression (which is implemented by default), Random Forest and Gradient Boosting.

# In[25]:


rf = ml.RandomForestClassifier(max_depth=5, n_estimators=20, random_state=ml.alg_random_state)
gb = ml.GradientBoostingClassifier(loss='log_loss', max_depth=3, learning_rate=0.1, random_state=ml.alg_random_state)


# ## Balanced Dataset

# #### Train Initial Models

# In[26]:


train_feature_sets_balanced = [{'name': 'style', 'features': style_train_balanced['features']}, {'name': 'word2vec', 'features': word2vec_train_balanced['features']}]
test_feature_sets_balanced = [{'name': 'style', 'features': style_test_balanced['features']}, {'name': 'word2vec', 'features': word2vec_test_balanced['features']}]


# In[27]:


get_ipython().run_cell_magic('time', '', "stacking_models_balanced = ml.train_models(train_feature_sets_balanced, style_train_balanced['target'])\n")


# ### Single-algorithm

# First, the stacking will be done only on the same algorithms with different feature sets, while also testing for different final_classifiers.

# In[28]:


results_stacking_balanced_single = pd.DataFrame()


# #### Logistic Regression

# In[29]:


get_ipython().run_cell_magic('time', '', "stacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced['target'], exclude_models=['dt', 'rf', 'gb', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced['target'], stacked_clf, exclude_models=['dt', 'rf', 'gb', 'nb'])\nresults_stacking_balanced_single = pd.concat([results_stacking_balanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced['target'], exclude_models=['lr', 'rf', 'gb', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced['target'], stacked_clf, exclude_models=['lr', 'rf', 'gb', 'nb'])\nresults_stacking_balanced_single = pd.concat([results_stacking_balanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced['target'], exclude_models=['lr', 'dt', 'gb', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced['target'], stacked_clf, exclude_models=['lr', 'dt', 'gb', 'nb'])\nresults_stacking_balanced_single = pd.concat([results_stacking_balanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced['target'], exclude_models=['lr', 'dt', 'rf', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced['target'], stacked_clf, exclude_models=['lr', 'dt', 'rf', 'nb'])\nresults_stacking_balanced_single = pd.concat([results_stacking_balanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced['target'], exclude_models=['lr', 'dt', 'rf', 'gb'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced['target'], stacked_clf, exclude_models=['lr', 'dt', 'rf', 'gb'])\nresults_stacking_balanced_single = pd.concat([results_stacking_balanced_single, stacked_preds['results']])\n")


# #### Random Forest

# In[30]:


get_ipython().run_cell_magic('time', '', "stacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced['target'], final_classifier=rf, exclude_models=['dt', 'rf', 'gb', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced['target'], stacked_clf, exclude_models=['dt', 'rf', 'gb', 'nb'])\nresults_stacking_balanced_single = pd.concat([results_stacking_balanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced['target'], final_classifier=rf, exclude_models=['lr', 'rf', 'gb', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced['target'], stacked_clf, exclude_models=['lr', 'rf', 'gb', 'nb'])\nresults_stacking_balanced_single = pd.concat([results_stacking_balanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced['target'], final_classifier=rf, exclude_models=['lr', 'dt', 'gb', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced['target'], stacked_clf, exclude_models=['lr', 'dt', 'gb', 'nb'])\nresults_stacking_balanced_single = pd.concat([results_stacking_balanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced['target'], final_classifier=rf, exclude_models=['lr', 'dt', 'rf', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced['target'], stacked_clf, exclude_models=['lr', 'dt', 'rf', 'nb'])\nresults_stacking_balanced_single = pd.concat([results_stacking_balanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced['target'], final_classifier=rf, exclude_models=['lr', 'dt', 'rf', 'gb'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced['target'], stacked_clf, exclude_models=['lr', 'dt', 'rf', 'gb'])\nresults_stacking_balanced_single = pd.concat([results_stacking_balanced_single, stacked_preds['results']])\n")


# #### Gradient Boosting

# In[31]:


get_ipython().run_cell_magic('time', '', "stacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced['target'], final_classifier=gb, exclude_models=['dt', 'rf', 'gb', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced['target'], stacked_clf, exclude_models=['dt', 'rf', 'gb', 'nb'])\nresults_stacking_balanced_single = pd.concat([results_stacking_balanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced['target'], final_classifier=gb, exclude_models=['lr', 'rf', 'gb', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced['target'], stacked_clf, exclude_models=['lr', 'rf', 'gb', 'nb'])\nresults_stacking_balanced_single = pd.concat([results_stacking_balanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced['target'], final_classifier=gb, exclude_models=['lr', 'dt', 'gb', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced['target'], stacked_clf, exclude_models=['lr', 'dt', 'gb', 'nb'])\nresults_stacking_balanced_single = pd.concat([results_stacking_balanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced['target'], final_classifier=gb, exclude_models=['lr', 'dt', 'rf', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced['target'], stacked_clf, exclude_models=['lr', 'dt', 'rf', 'nb'])\nresults_stacking_balanced_single = pd.concat([results_stacking_balanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced['target'], final_classifier=gb, exclude_models=['lr', 'dt', 'rf', 'gb'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced['target'], stacked_clf, exclude_models=['lr', 'dt', 'rf', 'gb'])\nresults_stacking_balanced_single = pd.concat([results_stacking_balanced_single, stacked_preds['results']])\n")


# In[32]:


results_stacking_balanced_single


# These results are on par with the baseline models, exept for Gradient Boosting that performed somewhat worse. Random Forest achieved better results. This is probably the result of overfitting.

# The 6 best performing models will be kept to compare with other stacking configurations and the complete single-algorithm results dataset will be archived.

# In[33]:


results_stacking_balanced_full = results_stacking_balanced_single.copy()


# In[34]:


results_stacking_balanced_best = results_stacking_balanced_single.sort_values(by=['F1 Score'], ascending = [False]).head(6)


# ### Multi-algorithm

# Of course, it is possible to also use the outputs of more than one classifier, on both feature sets.

# In[35]:


results_stacking_balanced_multi = pd.DataFrame()


# #### Logistic Regression

# In[36]:


get_ipython().run_cell_magic('time', '', 'stacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], exclude_models=[])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[], result_row_name="Algorithms: all, with LogisticRegression")\nresults_stacking_balanced_multi = pd.concat([results_stacking_balanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], exclude_models=[\'dt\', \'nb\'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'])\nresults_stacking_balanced_multi = pd.concat([results_stacking_balanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'lr\'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'])\nresults_stacking_balanced_multi = pd.concat([results_stacking_balanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'rf\'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'])\nresults_stacking_balanced_multi = pd.concat([results_stacking_balanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'gb\'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'])\nresults_stacking_balanced_multi = pd.concat([results_stacking_balanced_multi, stacked_preds[\'results\']])\n')


# #### Random Forest

# In[37]:


get_ipython().run_cell_magic('time', '', 'stacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[], result_row_name="Algorithms: all, with RandomForestClassifier")\nresults_stacking_balanced_multi = pd.concat([results_stacking_balanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'])\nresults_stacking_balanced_multi = pd.concat([results_stacking_balanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'lr\'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'])\nresults_stacking_balanced_multi = pd.concat([results_stacking_balanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'rf\'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'])\nresults_stacking_balanced_multi = pd.concat([results_stacking_balanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'gb\'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'])\nresults_stacking_balanced_multi = pd.concat([results_stacking_balanced_multi, stacked_preds[\'results\']])\n')


# #### Gradient Boosting

# In[38]:


get_ipython().run_cell_magic('time', '', 'stacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[], result_row_name="Algorithms: all, with GradientBoostingClassifier")\nresults_stacking_balanced_multi = pd.concat([results_stacking_balanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'])\nresults_stacking_balanced_multi = pd.concat([results_stacking_balanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'lr\'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'])\nresults_stacking_balanced_multi = pd.concat([results_stacking_balanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'rf\'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'])\nresults_stacking_balanced_multi = pd.concat([results_stacking_balanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'gb\'])\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'])\nresults_stacking_balanced_multi = pd.concat([results_stacking_balanced_multi, stacked_preds[\'results\']])\n')


# In[39]:


results_stacking_balanced_multi


# As expected, using more than one classifier consistently improves the classification accuracy for both LR and RF. The best level 1 classifier seems to be Random Forest, and Logistic Regression gives better results when used as a level 0 classifier. However, all combinations performed quite well, except for some time in Gradient Boosting.
# 
# On the other hand, Naive Bayes and Decision Tree classifiers do not affect the result at all or even reduce the accuracy so from now on they will be excluded in order to reduce the execution time.

# The top 10 models will be added to the best model results dataset.

# In[40]:


results_stacking_balanced_best = pd.concat([results_stacking_balanced_best, results_stacking_balanced_multi.sort_values(by=['F1 Score'], ascending = [False]).head(10)])


# In[41]:


results_stacking_balanced_full = pd.concat([results_stacking_balanced_full, results_stacking_balanced_multi])


# ### Appending all features

# Another variation of stacking includes appending the predictions to the other feature sets and then train the final classifier with all the features.<br>
# Since using both feature sets has been proven to improve accuracy on the baselines, the predictions will be appended to the merged feature set.

# In[42]:


results_stacking_balanced_append = pd.DataFrame()


# #### Logistic Regression

# In[43]:


get_ipython().run_cell_magic('time', '', 'stacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], exclude_models=[], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[], append_features=True, result_row_name="Algorithms: all, with LogisticRegression (with appended features)")\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], exclude_models=[\'dt\', \'nb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\n# Single level 0\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'gb\', \'rf\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\', \'rf\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'gb\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\', \'lr\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'rf\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\', \'lr\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n')


# #### Random Forest

# In[44]:


get_ipython().run_cell_magic('time', '', 'stacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[], append_features=True, result_row_name="Algorithms: all, with RandomForestClassifier (with appended features)")\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\n# Single level 0\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'gb\', \'rf\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\', \'rf\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'gb\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\', \'lr\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'rf\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\', \'lr\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n')


# #### Gradient Boosting

# In[45]:


get_ipython().run_cell_magic('time', '', 'stacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[], append_features=True, result_row_name="Algorithms: all, with GradientBoostingClassifier (with appended features)")\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\n# Single level 0\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'gb\', \'rf\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\', \'rf\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'gb\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\', \'lr\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_balanced, train_feature_sets_balanced, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'rf\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_balanced, test_feature_sets_balanced, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\', \'lr\'], append_features=True)\nresults_stacking_balanced_append = pd.concat([results_stacking_balanced_append, stacked_preds[\'results\']])\n')


# In[46]:


results_stacking_balanced_append


# Adding the initial feature sets to the final classifier seems to mostly harm performance on the balanced dataset. This is most likely due to overfitting, since the level 1 classifier becomes extremely specialized at recognizing the emails provided in the training set and fails to generalize for the test set.
# 
# However, when using Gradient Boosting as the final classifier, it manages at some cases to outperform the model without the appended features.

# The top 12 of these models will be added to the best result dataset, for comparison.

# In[47]:


results_stacking_balanced_best = pd.concat([results_stacking_balanced_best, results_stacking_balanced_append.sort_values(by=['F1 Score'], ascending = [False]).head(12)])


# In[48]:


results_stacking_balanced_full = pd.concat([results_stacking_balanced_full, results_stacking_balanced_append])


# ### Merged Classifiers

# Finally, for the sake of completeness, try stacking the level 0 classifiers that were trained with the merged dataset.

# In[49]:


train_feature_sets_balanced_merged = [{'name': 'merge', 'features': style_content_train_balanced}]
test_feature_sets_balanced_merged = [{'name': 'merge', 'features': style_content_test_balanced}]


# In[50]:


lr_merged_balanced = {'model' : lr_style_content_balanced, 'scaler': lr_style_content_balanced_scaler}
nb_merged_balanced = {'model' : nb_style_content_balanced, 'scaler': nb_style_content_balanced_scaler}

merged_models_balanced = [{'name' : 'lr', 'features' : 'merge', 'model' : lr_merged_balanced},
                          {'name' : 'dt', 'features' : 'merge', 'model' : dt_style_content_balanced},
                          {'name' : 'rf', 'features' : 'merge', 'model' : rf_style_content_balanced},
                          {'name' : 'gb', 'features' : 'merge', 'model' : gb_style_content_balanced},
                          {'name' : 'nb', 'features' : 'merge', 'model' : nb_merged_balanced}]


# In[51]:


results_stacking_balanced_merged = pd.DataFrame()


# #### Logistic Regression

# In[52]:


get_ipython().run_cell_magic('time', '', 'stacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], exclude_models=[\'dt\', \'nb\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'], append_features=False, result_row_name="Algorithms: lr, rf, gb merged, with LogisticRegression")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=False, result_row_name="Algorithms: rf, gb merged, with LogisticRegression")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=False, result_row_name="Algorithms: lr, gb merged, with LogisticRegression")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=False, result_row_name="Algorithms: rf, lr merged, with LogisticRegression")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\n# Append features\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], exclude_models=[\'dt\', \'nb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'], append_features=True, result_row_name="Algorithms: lr, rf, gb merged, with LogisticRegression (with appended features)")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True, result_row_name="Algorithms: rf, gb merged, with LogisticRegression (with appended features)")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True, result_row_name="Algorithms: lr, gb merged, with LogisticRegression (with appended features)")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True, result_row_name="Algorithms: rf, lr merged, with LogisticRegression (with appended features)")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n')


# #### Random Forest

# In[53]:


get_ipython().run_cell_magic('time', '', 'stacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'], append_features=False, result_row_name="Algorithms: lr, rf, gb merged, with RandomForestClassifier")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=False, result_row_name="Algorithms: rf, gb merged, with RandomForestClassifier")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=False, result_row_name="Algorithms: lr, gb merged, with RandomForestClassifier")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=False, result_row_name="Algorithms: rf, lr merged, with RandomForestClassifier")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\n# Append features\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'], append_features=True, result_row_name="Algorithms: lr, rf, gb merged, with RandomForestClassifier (with appended features)")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True, result_row_name="Algorithms: rf, gb merged, with RandomForestClassifier (with appended features)")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True, result_row_name="Algorithms: lr, gb merged, with RandomForestClassifier (with appended features)")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True, result_row_name="Algorithms: rf, lr merged, with RandomForestClassifier (with appended features)")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n')


# #### Gradient Boosting

# In[54]:


get_ipython().run_cell_magic('time', '', 'stacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'], append_features=False, result_row_name="Algorithms: lr, rf, gb merged, with GradientBoostingClassifier")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=False, result_row_name="Algorithms: rf, gb merged, with GradientBoostingClassifier")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=False, result_row_name="Algorithms: lr, gb merged, with GradientBoostingClassifier")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=False, result_row_name="Algorithms: rf, lr merged, with GradientBoostingClassifier")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\n# Append features\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'], append_features=True, result_row_name="Algorithms: lr, rf, gb merged, with GradientBoostingClassifier (with appended features)")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True, result_row_name="Algorithms: rf, gb merged, with GradientBoostingClassifier (with appended features)")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True, result_row_name="Algorithms: lr, gb merged, with GradientBoostingClassifier (with appended features)")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_balanced, train_feature_sets_balanced_merged, style_train_balanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_balanced, test_feature_sets_balanced_merged, style_test_balanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True, result_row_name="Algorithms: rf, lr merged, with GradientBoostingClassifier (with appended features)")\nresults_stacking_balanced_merged = pd.concat([results_stacking_balanced_merged, stacked_preds[\'results\']])\n')


# In[55]:


results_stacking_balanced_merged


# In general, the addition of the initial features on the level 1 classifier gives better results, but, while consistent, the results were not great. The results of the single level 0 classifiers were surprisingly good though for Logistic Regression and Gradient Boosting.
# 
# This is likely because the level 0 classifiers were more specialized compared to training on both feature sets separately.

# The top 13 results will be added to the dataset for comparison.

# In[56]:


results_stacking_balanced_best = pd.concat([results_stacking_balanced_best, results_stacking_balanced_merged.sort_values(by=['F1 Score'], ascending = [False]).head(10)])


# In[57]:


results_stacking_balanced_full = pd.concat([results_stacking_balanced_full, results_stacking_balanced_merged])


# In[58]:


results_stacking_balanced_best.sort_values(by=['F1 Score'], ascending = [False]).head(13)


# ## Imbalanced Dataset

# #### Train Initial Models

# In[59]:


train_feature_sets_imbalanced = [{'name': 'style', 'features': style_train_imbalanced['features']}, {'name': 'word2vec', 'features': word2vec_train_imbalanced['features']}]
test_feature_sets_imbalanced = [{'name': 'style', 'features': style_test_imbalanced['features']}, {'name': 'word2vec', 'features': word2vec_test_imbalanced['features']}]


# In[60]:


get_ipython().run_cell_magic('time', '', "stacking_models_imbalanced = ml.train_models(train_feature_sets_imbalanced, style_train_imbalanced['target'])\n")


# ### Single-algorithm

# In[61]:


results_stacking_imbalanced_single = pd.DataFrame()


# #### Logistic Regression

# In[62]:


get_ipython().run_cell_magic('time', '', "stacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced['target'], exclude_models=['dt', 'rf', 'gb', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced['target'], stacked_clf, exclude_models=['dt', 'rf', 'gb', 'nb'])\nresults_stacking_imbalanced_single = pd.concat([results_stacking_imbalanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced['target'], exclude_models=['lr', 'rf', 'gb', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced['target'], stacked_clf, exclude_models=['lr', 'rf', 'gb', 'nb'])\nresults_stacking_imbalanced_single = pd.concat([results_stacking_imbalanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced['target'], exclude_models=['lr', 'dt', 'gb', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced['target'], stacked_clf, exclude_models=['lr', 'dt', 'gb', 'nb'])\nresults_stacking_imbalanced_single = pd.concat([results_stacking_imbalanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced['target'], exclude_models=['lr', 'dt', 'rf', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced['target'], stacked_clf, exclude_models=['lr', 'dt', 'rf', 'nb'])\nresults_stacking_imbalanced_single = pd.concat([results_stacking_imbalanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced['target'], exclude_models=['lr', 'dt', 'rf', 'gb'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced['target'], stacked_clf, exclude_models=['lr', 'dt', 'rf', 'gb'])\nresults_stacking_imbalanced_single = pd.concat([results_stacking_imbalanced_single, stacked_preds['results']])\n")


# #### Random Forest

# In[63]:


get_ipython().run_cell_magic('time', '', "stacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced['target'], final_classifier=rf, exclude_models=['dt', 'rf', 'gb', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced['target'], stacked_clf, exclude_models=['dt', 'rf', 'gb', 'nb'])\nresults_stacking_imbalanced_single = pd.concat([results_stacking_imbalanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced['target'], final_classifier=rf, exclude_models=['lr', 'rf', 'gb', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced['target'], stacked_clf, exclude_models=['lr', 'rf', 'gb', 'nb'])\nresults_stacking_imbalanced_single = pd.concat([results_stacking_imbalanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced['target'], final_classifier=rf, exclude_models=['lr', 'dt', 'gb', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced['target'], stacked_clf, exclude_models=['lr', 'dt', 'gb', 'nb'])\nresults_stacking_imbalanced_single = pd.concat([results_stacking_imbalanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced['target'], final_classifier=rf, exclude_models=['lr', 'dt', 'rf', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced['target'], stacked_clf, exclude_models=['lr', 'dt', 'rf', 'nb'])\nresults_stacking_imbalanced_single = pd.concat([results_stacking_imbalanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced['target'], final_classifier=rf, exclude_models=['lr', 'dt', 'rf', 'gb'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced['target'], stacked_clf, exclude_models=['lr', 'dt', 'rf', 'gb'])\nresults_stacking_imbalanced_single = pd.concat([results_stacking_imbalanced_single, stacked_preds['results']])\n")


# #### Gradient Boosting

# In[64]:


get_ipython().run_cell_magic('time', '', "stacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced['target'], final_classifier=gb, exclude_models=['dt', 'rf', 'gb', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced['target'], stacked_clf, exclude_models=['dt', 'rf', 'gb', 'nb'])\nresults_stacking_imbalanced_single = pd.concat([results_stacking_imbalanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced['target'], final_classifier=gb, exclude_models=['lr', 'rf', 'gb', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced['target'], stacked_clf, exclude_models=['lr', 'rf', 'gb', 'nb'])\nresults_stacking_imbalanced_single = pd.concat([results_stacking_imbalanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced['target'], final_classifier=gb, exclude_models=['lr', 'dt', 'gb', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced['target'], stacked_clf, exclude_models=['lr', 'dt', 'gb', 'nb'])\nresults_stacking_imbalanced_single = pd.concat([results_stacking_imbalanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced['target'], final_classifier=gb, exclude_models=['lr', 'dt', 'rf', 'nb'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced['target'], stacked_clf, exclude_models=['lr', 'dt', 'rf', 'nb'])\nresults_stacking_imbalanced_single = pd.concat([results_stacking_imbalanced_single, stacked_preds['results']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced['target'], final_classifier=gb, exclude_models=['lr', 'dt', 'rf', 'gb'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced['target'], stacked_clf, exclude_models=['lr', 'dt', 'rf', 'gb'])\nresults_stacking_imbalanced_single = pd.concat([results_stacking_imbalanced_single, stacked_preds['results']])\n")


# In[65]:


results_stacking_imbalanced_single


# The results were somewhat consistent and slightly better at some cases with the results of the imbalanced dataset. Random Forest showed significant improvement and even NB managed to classify something. Compared to the merged feature sets, RF performed significantly better and GB too managed to outperformed it.

# In[66]:


results_stacking_imbalanced_full = results_stacking_imbalanced_single.copy()


# In[67]:


results_stacking_imbalanced_best = results_stacking_imbalanced_single.sort_values(by=['F1 Score'], ascending = [False]).head(6)


# ### Multi-algorithm

# In[68]:


results_stacking_imbalanced_multi = pd.DataFrame()


# #### Logistic Regression

# In[69]:


get_ipython().run_cell_magic('time', '', 'stacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], exclude_models=[])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[], result_row_name="Algorithms: all, with LogisticRegression")\nresults_stacking_imbalanced_multi = pd.concat([results_stacking_imbalanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], exclude_models=[\'dt\', \'nb\'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'])\nresults_stacking_imbalanced_multi = pd.concat([results_stacking_imbalanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'lr\'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'])\nresults_stacking_imbalanced_multi = pd.concat([results_stacking_imbalanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'rf\'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'])\nresults_stacking_imbalanced_multi = pd.concat([results_stacking_imbalanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'gb\'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'])\nresults_stacking_imbalanced_multi = pd.concat([results_stacking_imbalanced_multi, stacked_preds[\'results\']])\n')


# #### Random Forest

# In[70]:


get_ipython().run_cell_magic('time', '', 'stacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[], result_row_name="Algorithms: all, with RandomForestClassifier")\nresults_stacking_imbalanced_multi = pd.concat([results_stacking_imbalanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'])\nresults_stacking_imbalanced_multi = pd.concat([results_stacking_imbalanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'lr\'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'])\nresults_stacking_imbalanced_multi = pd.concat([results_stacking_imbalanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'rf\'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'])\nresults_stacking_imbalanced_multi = pd.concat([results_stacking_imbalanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'gb\'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'])\nresults_stacking_imbalanced_multi = pd.concat([results_stacking_imbalanced_multi, stacked_preds[\'results\']])\n')


# #### Gradient Boosting

# In[71]:


get_ipython().run_cell_magic('time', '', 'stacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[], result_row_name="Algorithms: all, with GradientBoostingClassifier")\nresults_stacking_imbalanced_multi = pd.concat([results_stacking_imbalanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'])\nresults_stacking_imbalanced_multi = pd.concat([results_stacking_imbalanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'lr\'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'])\nresults_stacking_imbalanced_multi = pd.concat([results_stacking_imbalanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'rf\'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'])\nresults_stacking_imbalanced_multi = pd.concat([results_stacking_imbalanced_multi, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'gb\'])\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'])\nresults_stacking_imbalanced_multi = pd.concat([results_stacking_imbalanced_multi, stacked_preds[\'results\']])\n')


# In[72]:


results_stacking_imbalanced_multi


# Of course, these models performed better on average than the stacking only of different feature sets. On the imbalanced dataset, Naive Bayes and Decision Tree did have more impact on the prediction accuracy.

# In[73]:


results_stacking_imbalanced_best = pd.concat([results_stacking_imbalanced_best, results_stacking_imbalanced_multi.sort_values(by=['F1 Score'], ascending = [False]).head(13)])


# In[74]:


results_stacking_imbalanced_full = pd.concat([results_stacking_imbalanced_full, results_stacking_imbalanced_multi])


# ### Appending all features

# In[75]:


results_stacking_imbalanced_append = pd.DataFrame()


# #### Logistic Regression

# In[76]:


get_ipython().run_cell_magic('time', '', 'stacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], exclude_models=[], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[], append_features=True, result_row_name="Algorithms: all, with LogisticRegression (with appended features)")\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], exclude_models=[\'dt\', \'nb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\n# Single level 0\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'gb\', \'rf\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\', \'rf\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'gb\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\', \'lr\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'rf\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\', \'lr\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n')


# #### Random Forest

# In[77]:


get_ipython().run_cell_magic('time', '', 'stacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[], append_features=True, result_row_name="Algorithms: all, with RandomForestClassifier (with appended features)")\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\n# Single level 0\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'gb\', \'rf\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\', \'rf\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'gb\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\', \'lr\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'rf\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\', \'lr\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n')


# #### Gradient Boosting

# In[78]:


get_ipython().run_cell_magic('time', '', 'stacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[], append_features=True, result_row_name="Algorithms: all, with GradientBoostingClassifier (with appended features)")\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\n# Single level 0\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'gb\', \'rf\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\', \'rf\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'gb\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\', \'lr\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(stacking_models_imbalanced, train_feature_sets_imbalanced, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'rf\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(stacking_models_imbalanced, test_feature_sets_imbalanced, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\', \'lr\'], append_features=True)\nresults_stacking_imbalanced_append = pd.concat([results_stacking_imbalanced_append, stacked_preds[\'results\']])\n')


# In[79]:


results_stacking_imbalanced_append


# Adding the initial feature sets to the final classifier also mostly harms performance on the imbalanced dataset. The best performing model now only barely performed better than without the features. The algorithm that performed better was Gradient Boosting. Also, there were some single-classifier models in the top performing ones.

# In[80]:


results_stacking_imbalanced_best = pd.concat([results_stacking_imbalanced_best, results_stacking_imbalanced_append.sort_values(by=['F1 Score'], ascending = [False]).head(12)])


# In[81]:


results_stacking_imbalanced_full = pd.concat([results_stacking_imbalanced_full, results_stacking_imbalanced_append])


# ### Merged Classifiers

# In[82]:


train_feature_sets_imbalanced_merged = [{'name': 'merge', 'features': style_content_train_imbalanced}]
test_feature_sets_imbalanced_merged = [{'name': 'merge', 'features': style_content_test_imbalanced}]


# In[83]:


lr_merged_imbalanced = {'model' : lr_style_content_imbalanced, 'scaler': lr_style_content_imbalanced_scaler}
nb_merged_imbalanced = {'model' : nb_style_content_imbalanced, 'scaler': nb_style_content_imbalanced_scaler}

merged_models_imbalanced = [{'name' : 'lr', 'features' : 'merge', 'model' : lr_merged_imbalanced},
                          {'name' : 'dt', 'features' : 'merge', 'model' : dt_style_content_imbalanced},
                          {'name' : 'rf', 'features' : 'merge', 'model' : rf_style_content_imbalanced},
                          {'name' : 'gb', 'features' : 'merge', 'model' : gb_style_content_imbalanced},
                          {'name' : 'nb', 'features' : 'merge', 'model' : nb_merged_imbalanced}]


# In[84]:


results_stacking_imbalanced_merged = pd.DataFrame()


# #### Logistic Regression

# In[85]:


get_ipython().run_cell_magic('time', '', 'stacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], exclude_models=[\'dt\', \'nb\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'], append_features=False, result_row_name="Algorithms: lr, rf, gb merged, with LogisticRegression")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=False, result_row_name="Algorithms: rf, gb merged, with LogisticRegression")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=False, result_row_name="Algorithms: lr, gb merged, with LogisticRegression")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=False, result_row_name="Algorithms: rf, lr merged, with LogisticRegression")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\n# Append features\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], exclude_models=[\'dt\', \'nb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'], append_features=True, result_row_name="Algorithms: lr, rf, gb merged, with LogisticRegression (with appended features)")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True, result_row_name="Algorithms: rf, gb merged, with LogisticRegression (with appended features)")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True, result_row_name="Algorithms: lr, gb merged, with LogisticRegression (with appended features)")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True, result_row_name="Algorithms: rf, lr merged, with LogisticRegression (with appended features)")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n')


# #### Random Forest

# In[86]:


get_ipython().run_cell_magic('time', '', 'stacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'], append_features=False, result_row_name="Algorithms: lr, rf, gb merged, with RandomForestClassifier")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=False, result_row_name="Algorithms: rf, gb merged, with RandomForestClassifier")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=False, result_row_name="Algorithms: lr, gb merged, with RandomForestClassifier")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=False, result_row_name="Algorithms: rf, lr merged, with RandomForestClassifier")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\n# Append features\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'], append_features=True, result_row_name="Algorithms: lr, rf, gb merged, with RandomForestClassifier (with appended features)")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True, result_row_name="Algorithms: rf, gb merged, with RandomForestClassifier (with appended features)")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True, result_row_name="Algorithms: lr, gb merged, with RandomForestClassifier (with appended features)")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], final_classifier=rf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True, result_row_name="Algorithms: rf, lr merged, with RandomForestClassifier (with appended features)")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n')


# #### Gradient Boosting

# In[87]:


get_ipython().run_cell_magic('time', '', 'stacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'], append_features=False, result_row_name="Algorithms: lr, rf, gb merged, with GradientBoostingClassifier")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=False, result_row_name="Algorithms: rf, gb merged, with GradientBoostingClassifier")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=False, result_row_name="Algorithms: lr, gb merged, with GradientBoostingClassifier")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=False)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=False, result_row_name="Algorithms: rf, lr merged, with GradientBoostingClassifier")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\n# Append features\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\'], append_features=True, result_row_name="Algorithms: lr, rf, gb merged, with GradientBoostingClassifier (with appended features)")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'lr\'], append_features=True, result_row_name="Algorithms: rf, gb merged, with GradientBoostingClassifier (with appended features)")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'rf\'], append_features=True, result_row_name="Algorithms: lr, gb merged, with GradientBoostingClassifier (with appended features)")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n\nstacked_clf = ml.train_stacked_models(merged_models_imbalanced, train_feature_sets_imbalanced_merged, style_train_imbalanced[\'target\'], final_classifier=gb, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True)\nstacked_preds = ml.test_stacked_models(merged_models_imbalanced, test_feature_sets_imbalanced_merged, style_test_imbalanced[\'target\'], stacked_clf, exclude_models=[\'dt\', \'nb\', \'gb\'], append_features=True, result_row_name="Algorithms: rf, lr merged, with GradientBoostingClassifier (with appended features)")\nresults_stacking_imbalanced_merged = pd.concat([results_stacking_imbalanced_merged, stacked_preds[\'results\']])\n')


# In[88]:


results_stacking_imbalanced_merged


# These models did not perform as well as the previous ones, but in general were better than the baseline with merged features.

# In[89]:


results_stacking_imbalanced_best = pd.concat([results_stacking_imbalanced_best, results_stacking_imbalanced_merged.sort_values(by=['F1 Score'], ascending = [False]).head(10)])


# In[90]:


results_stacking_imbalanced_full = pd.concat([results_stacking_imbalanced_full, results_stacking_imbalanced_merged])


# In[91]:


results_stacking_imbalanced_best.sort_values(by=['F1 Score'], ascending = [False]).head(13)

