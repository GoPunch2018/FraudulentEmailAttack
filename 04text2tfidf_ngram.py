import os
import pandas as pd
import numpy as np

pd.options.display.max_columns = 250
pd.options.display.max_colwidth = 160

import features as util
from raw_utils import save_to_csv
from preprocessing import dataset_add_columns

from ast import literal_eval

cwd = os.getcwd()
csv_path = os.path.join(cwd, 'data/csv/')

generic_spam_tokens = 'generic_spam_tokens.csv'
non_targeted_tokens = 'non_targeted_tokens.csv'

generic_spam_tokens = pd.read_csv(os.path.join(csv_path, generic_spam_tokens), index_col=0,
                                  converters={'body': literal_eval})
non_targeted_tokens = pd.read_csv(os.path.join(csv_path, non_targeted_tokens), index_col=0,
                                  converters={'body': literal_eval})

tfidf_balanced = util.tfidf_features_unsupervised(generic_spam_tokens['body'], min_df=5, ngram_range=(1, 3),
                                                  max_features=500, topic_number=10)
W_generic_spam = tfidf_balanced['document-topic']
H_generic_spam = tfidf_balanced['topic-term']

tfidf_non_targeted = util.tfidf_features_unsupervised(non_targeted_tokens['body'], min_df=5, ngram_range=(1, 3),
                                                      max_features=500, topic_number=15)
W_non_targeted = tfidf_non_targeted['document-topic']
H_non_targeted = tfidf_non_targeted['topic-term']

topic_dimension = util.topic_dimension_mapping()

document_dimension = np.dot(W_non_targeted, topic_dimension)

# 将结果转换回 DataFrame （如果需要）
result_df = pd.DataFrame(document_dimension, columns=['Dimension1', 'Dimension2', ..., 'DimensionN'])
word2vec_balanced = util.word2vec_features(generic_spam_tokens['body'], non_targeted_tokens['body'],
                                           vector_size=100, min_count=5)

# In[12]:


word2vec_train_balanced = word2vec_balanced['word2vec_train']
word2vec_test_balanced = word2vec_balanced['word2vec_test']
word2vec_model_balanced = word2vec_balanced['vectorizer']

# In[13]:


word2vec_imbalanced = util.word2vec_features(train_imbalanced_tokens['body'], test_imbalanced_tokens['body'],
                                             vector_size=100, min_count=5)

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


selected_tfidf_balanced = util.chi2_feature_selection(tfidf_train_balanced, generic_spam_tokens['class'],
                                                      tfidf_test_balanced, percentile=50)

# In[17]:


tfidf_sel_train_balanced = selected_tfidf_balanced['features_train']
tfidf_sel_test_balanced = selected_tfidf_balanced['features_test']
tfidf_sel_model_balanced = selected_tfidf_balanced['selector']

# In[18]:


selected_tfidf_imbalanced = util.chi2_feature_selection(tfidf_train_imbalanced, train_imbalanced_tokens['class'],
                                                        tfidf_test_imbalanced, percentile=50)

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


column_names = ['email_class', 'email_id']  # column names changed in case the word class or id appear in the token list

# ### TF-IDF

# In[22]:


final_tfidf_train_balanced = dataset_add_columns(tfidf_sel_train_balanced,
                                                 [generic_spam_tokens['class'], generic_spam_tokens['id']],
                                                 column_names)
final_tfidf_test_balanced = dataset_add_columns(tfidf_sel_test_balanced,
                                                [non_targeted_spam_tokens['class'], non_targeted_spam_tokens['id']],
                                                column_names)

# In[23]:


final_tfidf_train_imbalanced = dataset_add_columns(tfidf_sel_train_imbalanced,
                                                   [train_imbalanced_tokens['class'], train_imbalanced_tokens['id']],
                                                   column_names)
final_tfidf_test_imbalanced = dataset_add_columns(tfidf_sel_test_imbalanced,
                                                  [test_imbalanced_tokens['class'], test_imbalanced_tokens['id']],
                                                  column_names)

# Looking into one of the previously explored examples:

# In[24]:


final_tfidf_train_balanced[final_tfidf_train_balanced['email_id'] == 6]

# The words that appear more in the email have a bigger score, while the words that don't appear at all have a score of zero.

# ### Word2Vec

# In[25]:


final_word2vec_train_balanced = dataset_add_columns(word2vec_train_balanced,
                                                    [generic_spam_tokens['class'], generic_spam_tokens['id']],
                                                    column_names)
final_word2vec_test_balanced = dataset_add_columns(word2vec_test_balanced,
                                                   [non_targeted_spam_tokens['class'], non_targeted_spam_tokens['id']],
                                                   column_names)

# In[26]:


final_word2vec_train_imbalanced = dataset_add_columns(word2vec_train_imbalanced,
                                                      [train_imbalanced_tokens['class'], train_imbalanced_tokens['id']],
                                                      column_names)
final_word2vec_test_imbalanced = dataset_add_columns(word2vec_test_imbalanced,
                                                     [test_imbalanced_tokens['class'], test_imbalanced_tokens['id']],
                                                     column_names)

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
