#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
pd.options.display.max_columns = 250
pd.options.display.max_colwidth = 160

import features as util
from raw_utils import save_to_csv
from preprocessing import dataset_add_columns

from ast import literal_eval


# ### Read Data

# In[2]:


# Path
cwd = os.getcwd()
csv_path = os.path.join(cwd, 'data/csv/')

train_tokens = ['train_balanced_tokens.csv', 'train_imbalanced_tokens.csv']
test_tokens = ['test_balanced_tokens.csv', 'test_imbalanced_tokens.csv']


# #### Tokenized emails

# In[3]:


train_balanced_tokens = pd.read_csv(os.path.join(csv_path, train_tokens[0]), index_col=0, converters={'body': literal_eval})
test_balanced_tokens = pd.read_csv(os.path.join(csv_path, test_tokens[0]), index_col=0, converters={'body': literal_eval})


# In[4]:


train_imbalanced_tokens = pd.read_csv(os.path.join(csv_path, train_tokens[1]), index_col=0, converters={'body': literal_eval})
test_imbalanced_tokens = pd.read_csv(os.path.join(csv_path, test_tokens[1]), index_col=0, converters={'body': literal_eval})


# After the preprocessing, the data look like this:

# In[5]:


train_balanced_tokens.head()


# # Feature Extraction

# Before inputing the emails to the machine learning algorithms, they have to be converted to numberical matrices.<br>
# This process is called **feature extraction**. Different methods of achieving this will be tried, in order to compare their results.

# ## Text Vectorization

# The baseline feature set will simply consist of numerical representations of the text data. This process is also called **vectorization**. 

# ### TF-IDF

# One of the most basic ways is to calculate the **tf-idf** (term frequency-inverse document frequency) score of the emails.<br>
# In order to have a lower dimensionality and since not all words from the corpus will be of importance, only the top 500 most frequent terms are used.

# In[6]:


tfidf_balanced = util.tfidf_features(train_balanced_tokens['body'], test_balanced_tokens['body'], min_df=5, max_features=500)


# In[7]:


tfidf_train_balanced = tfidf_balanced['tfidf_train']
tfidf_test_balanced = tfidf_balanced['tfidf_test']
tfidf_model_balanced = tfidf_balanced['vectorizer']


# In[8]:


tfidf_imbalanced = util.tfidf_features(train_imbalanced_tokens['body'], test_imbalanced_tokens['body'], min_df=5, max_features=500)


# In[9]:


tfidf_train_imbalanced = tfidf_imbalanced['tfidf_train']
tfidf_test_imbalanced = tfidf_imbalanced['tfidf_test']
tfidf_model_imbalanced = tfidf_imbalanced['vectorizer']


# As an example, here is a part of the calcuated matrix for the balanced train set:

# In[10]:


tfidf_train_balanced.head()


# ### Word2Vec

# A more advanced technique is **Word Embedding**, which calculates a high-dimensional vector for each word based on the probability distribution of this word appearing before or after another. In other words, words belonging to the same context usually appear close to each other in the corpus, so they will be closer in the vector space as well.<br>
# The chosen implementation is **Word2Vec**.

# After the word vectors are calculated, the vectors of each word in an email are being averaged, thus resulting in a single vector for each email.

# In[11]:


word2vec_balanced = util.word2vec_features(train_balanced_tokens['body'], test_balanced_tokens['body'], vector_size=100, min_count=5)


# In[12]:


word2vec_train_balanced = word2vec_balanced['word2vec_train']
word2vec_test_balanced = word2vec_balanced['word2vec_test']
word2vec_model_balanced = word2vec_balanced['vectorizer']


# In[13]:


word2vec_imbalanced = util.word2vec_features(train_imbalanced_tokens['body'], test_imbalanced_tokens['body'], vector_size=100, min_count=5)


# In[14]:


word2vec_train_imbalanced = word2vec_imbalanced['word2vec_train']
word2vec_test_imbalanced = word2vec_imbalanced['word2vec_test']
word2vec_model_imbalanced = word2vec_imbalanced['vectorizer']


# The resulting feature sets are like the following:

# In[15]:


word2vec_train_balanced.head()


# It should be noted that in this case, the columns do not provide information similar to how a tf-idf column corresponds to one word. This representation is purely for convenience and consistency, it won't matter during the prediction step.

# # Feature Selection

# In order to further reduce the dimensions of the feature matrix, the number of selected features will be halved using the top features according to the **chi-squared** feature selection method.

# ## Vectorization Features

# ### TF-IDF

# In[16]:


selected_tfidf_balanced = util.chi2_feature_selection(tfidf_train_balanced, train_balanced_tokens['class'], tfidf_test_balanced, percentile=50)


# In[17]:


tfidf_sel_train_balanced = selected_tfidf_balanced['features_train']
tfidf_sel_test_balanced = selected_tfidf_balanced['features_test']
tfidf_sel_model_balanced = selected_tfidf_balanced['selector']


# In[18]:


selected_tfidf_imbalanced = util.chi2_feature_selection(tfidf_train_imbalanced, train_imbalanced_tokens['class'], tfidf_test_imbalanced, percentile=50)


# In[19]:


tfidf_sel_train_imbalanced = selected_tfidf_imbalanced['features_train']
tfidf_sel_test_imbalanced = selected_tfidf_imbalanced['features_test']
tfidf_sel_model_imbalanced = selected_tfidf_imbalanced['selector']


# The now-reduced train set:

# In[20]:


tfidf_sel_train_balanced.head()


# # Final Dataset Creation

# Before using the features for classification with the machine learning algorithms, it is best to tidy up the datasets and keep them consistent by concatenating the features, the id and the class columns in the same DataFrame.

# In[21]:


column_names = ['email_class', 'email_id'] # column names changed in case the word class or id appear in the token list


# ### TF-IDF

# In[22]:


final_tfidf_train_balanced = dataset_add_columns(tfidf_sel_train_balanced, [train_balanced_tokens['class'], train_balanced_tokens['id']], column_names)
final_tfidf_test_balanced = dataset_add_columns(tfidf_sel_test_balanced, [test_balanced_tokens['class'], test_balanced_tokens['id']], column_names)


# In[23]:


final_tfidf_train_imbalanced = dataset_add_columns(tfidf_sel_train_imbalanced, [train_imbalanced_tokens['class'], train_imbalanced_tokens['id']], column_names)
final_tfidf_test_imbalanced = dataset_add_columns(tfidf_sel_test_imbalanced, [test_imbalanced_tokens['class'], test_imbalanced_tokens['id']], column_names)


# Looking into one of the previously explored examples:

# In[24]:


final_tfidf_train_balanced[final_tfidf_train_balanced['email_id'] == 6]


# The words that appear more in the email have a bigger score, while the words that don't appear at all have a score of zero.

# ### Word2Vec

# In[25]:


final_word2vec_train_balanced = dataset_add_columns(word2vec_train_balanced, [train_balanced_tokens['class'], train_balanced_tokens['id']], column_names)
final_word2vec_test_balanced = dataset_add_columns(word2vec_test_balanced, [test_balanced_tokens['class'], test_balanced_tokens['id']], column_names)


# In[26]:


final_word2vec_train_imbalanced = dataset_add_columns(word2vec_train_imbalanced, [train_imbalanced_tokens['class'], train_imbalanced_tokens['id']], column_names)
final_word2vec_test_imbalanced = dataset_add_columns(word2vec_test_imbalanced, [test_imbalanced_tokens['class'], test_imbalanced_tokens['id']], column_names)


# In[27]:


final_tfidf_train_balanced.head()


# ### Saving the Results

# In[28]:


save_to_csv(final_tfidf_train_balanced, csv_path, 'tfidf_chi2_train_balanced.csv')
save_to_csv(final_tfidf_test_balanced, csv_path, 'tfidf_chi2_test_balanced.csv')

save_to_csv(final_tfidf_train_imbalanced, csv_path, 'tfidf_chi2_train_imbalanced.csv')
save_to_csv(final_tfidf_test_imbalanced, csv_path, 'tfidf_chi2_test_imbalanced.csv')


# In[29]:


save_to_csv(final_word2vec_train_balanced, csv_path, 'word2vec_train_balanced.csv')
save_to_csv(final_word2vec_test_balanced, csv_path, 'word2vec_test_balanced.csv')

save_to_csv(final_word2vec_train_imbalanced, csv_path, 'word2vec_train_imbalanced.csv')
save_to_csv(final_word2vec_test_imbalanced, csv_path, 'word2vec_test_imbalanced.csv')

