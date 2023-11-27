#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
pd.options.display.max_colwidth = 160

import preprocessing as util
from raw_utils import save_to_csv


# In[2]:


# Path
cwd = os.getcwd()
csv_path = os.path.join(cwd, 'data/csv/')

data_files = ['balanced.csv', 'imbalanced.csv']


# In[3]:


balanced = pd.read_csv(os.path.join(csv_path, data_files[0]), index_col=0, dtype={'body': 'object', 'class': 'bool', 'id': 'int16'})
balanced.info()


# In[4]:


imbalanced = pd.read_csv(os.path.join(csv_path, data_files[1]), index_col=0, dtype={'body': 'object', 'class': 'bool', 'id': 'int16'})
imbalanced.info()


# ### Initial Data

# This is the initial state of the data, along with some representative examples of phishing and legitimate emails.

# In[5]:


balanced.head(20)


# #### Legitimate emails:

# In[6]:


print(balanced['body'].iloc[1])


# In[7]:


print(balanced['body'].iloc[16])


# In[8]:


print(balanced['body'].iloc[18])


# #### Phishing Emails:

# In[9]:


print(balanced['body'].iloc[6])


# In[10]:


print(balanced['body'].iloc[11])


# In[11]:


print(balanced['body'].iloc[17])


# In[12]:


print(balanced['body'].iloc[3240])


# Some observations:
# - There is a lot of extra whitespace that can be sanitized later.
# - There still exist some emails with duplicated text, mostly because of the way the links from \<a\> tags were extracted.
# - Emails and URLs can give away the class of the message (domain enron.com vs domain monkey.org), so removing them should make the model more general.

# # Preprocessing

# We need to convert the text data into a format more suitable for use with machine learning algorithms.<br>
# Since we aim for two different feature sets, the process will be split.

# ## Basic Preprocessing

# These processeses should happen to all the feature sets.

# ### Replacing addresses

# As is obvious from the examples, a lot of the emails contain either **web addresses** (URLs) or **email addresses** that need to be removed in order for the frequency of certain domains to not influence the results.<br>
# In order for this information to not get completely lost however, those addresses will be replaced by the strings `'<urladdress>'` and `'<emailaddress>'` respectively. Those strings are chosen because they do not occur normally in the emails.

# In[13]:


balanced['body'] = balanced['body'].apply(util.replace_email)
balanced['body'] = balanced['body'].apply(util.replace_url)


# In[14]:


imbalanced['body'] = imbalanced['body'].apply(util.replace_email)
imbalanced['body'] = imbalanced['body'].apply(util.replace_url)


# In[15]:


print(balanced['body'].iloc[6])


# In[16]:


print(balanced['body'].iloc[17])


# The examples show that the URLs and email addresses have indeed been anonymized now.

# ## Preprocessing for content features

# This preprocessing is necessary in order to convert the text strings to lists of words, that will be vectorized in order to be used by machine learning algorithms.

# In[17]:


balanced_tokens = balanced.copy()
imbalanced_tokens = imbalanced.copy()


# ### Tokenization and stopword removal

# Tokenization is the process of splitting text into individual words. This is useful because generally speaking, the meaning of the text can easily be interpreted by analyzing the words present in the text.<br>
# Along with this process, letters are also converted to lowercase and punctuation or other special characters are removed.<br>
# Since there are some words (called **stopwords**) that do not contribute very much in meaning (like pronouns or simple verbs), they can be removed to reduce the noise.

# In[18]:


balanced_tokens['body'] = balanced_tokens['body'].apply(util.tokenize)
balanced_tokens['body'] = balanced_tokens['body'].apply(util.remove_stopwords)


# In[19]:


imbalanced_tokens['body'] = imbalanced_tokens['body'].apply(util.tokenize)
imbalanced_tokens['body'] = imbalanced_tokens['body'].apply(util.remove_stopwords)


# In[20]:


print(balanced_tokens['body'].iloc[6])


# In[21]:


print(balanced_tokens['body'].iloc[17])


# The example shows how a quite big chunk of text was reduced to a smaller list that contains the more meaningful words. The addresses still exist as tokens ('urladdress').<br>
# Also, some emails with duplicate emails obviously will have duplicate tokens, this however is not a big issue with most vectorizers.

# ### Lemmatization with POS tagging

# Lemmatization is the process that reduces the inflectional forms of a word to keep its root form. This is useful because the set of words that results from this process is smaller because all the inflections of a word are converted to one, thus reducing the dimensionality without sacrificing information.<br>
# In order to facilitate and improve the lemmatization, the **part-of-speech tagging** technique has been used. The POS of the word (which indicates whether a word is a noun, a verb, an adjective, or an adverb) is used as a part of the process.

# In[22]:


balanced_tokens['body'] = balanced_tokens['body'].apply(util.lemmatize)


# In[23]:


imbalanced_tokens['body'] = imbalanced_tokens['body'].apply(util.lemmatize)


# In[24]:


print(balanced_tokens['body'].iloc[6])


# The example shows how the lemmatization process has worked: words like 'holding' have been converted to their root form 'hold'.<br>
# In addition, it also shows the working of the POS tagging process, since the word 'incoming' has remained the same as it is used as an adjective and not as a verb.

# ## Preprocessing for style features

# This preprocessing is necessary in order to sanitize the raw email text and remove parsing artifacts, so that the stylometric features work better.

# In[25]:


balanced_text = balanced.copy()
imbalanced_text = imbalanced.copy()


# ### Whitespace Sanitization

# The first task should be stripping away any leading and trailing whitespace. In addition, newlines that only contain a dot are most likely artifacts from the parsing of HTML and can thus be removed with this dot placed at the previous line.

# In[26]:


balanced_text['body'] = balanced_text['body'].apply(util.sanitize_whitespace)


# In[27]:


imbalanced_text['body'] = imbalanced_text['body'].apply(util.sanitize_whitespace)


# ### Address Sanitization

# There are also artifacts from the URL/email anonymization that while innocent with tokenized texts, they are more harmful when the number of special characters in the text matters.

# In[28]:


balanced_text['body'] = balanced_text['body'].apply(util.sanitize_addresses)


# In[29]:


imbalanced_text['body'] = imbalanced_text['body'].apply(util.sanitize_addresses)


# One of the previous examples looks better now.

# In[30]:


print(balanced_text['body'].iloc[6])


# ## Deleting Empty Rows

# After all the preprocessing, it is possible that some of the emails are now empty (because they did not contain any useful words from the beginning).<br>
# So, these have to be removed to keep the data clean.

# In[31]:


balanced_tokens = balanced_tokens[balanced_tokens['body'].astype(bool)]
balanced_tokens.info()


# In[32]:


imbalanced_tokens = imbalanced_tokens[imbalanced_tokens['body'].astype(bool)]
imbalanced_tokens.info()


# In[33]:


balanced_text = balanced_text[balanced_text['body'].astype(bool)]
balanced_text.info()


# In[34]:


imbalanced_text = imbalanced_text[imbalanced_text['body'].astype(bool)]
imbalanced_text.info()


# In order to have the same emails in both feature sets, the text dataset will be filtered according to the tokenized one.

# In[35]:


balanced_text = balanced_text[balanced_text['id'].isin(balanced_tokens['id'])]
balanced_text.info()


# In[36]:


imbalanced_text = imbalanced_text[imbalanced_text['id'].isin(imbalanced_tokens['id'])]
imbalanced_text.info()


# Check for any discrepancies:

# In[37]:


(balanced_text['id'] != balanced_tokens['id']).any() and (imbalanced_text['id'] != imbalanced_tokens['id']).any()


# ## Train-Test Split

# In order to evaluate the classification process, only 80% of the data will be used to train the models. The remaining 20%, which will be unknown to the algorithms, will be used to test the performance of the classifiers on unknown data.

# ### Tokens

# In[38]:


train_balanced_tokens, test_balanced_tokens = util.dataset_split(balanced_tokens, percent=20)


# In[39]:


train_imbalanced_tokens, test_imbalanced_tokens = util.dataset_split(imbalanced_tokens, percent=20)


# In[40]:


train_balanced_tokens[train_balanced_tokens['id'] == 6]


# In[41]:


test_balanced_tokens[test_balanced_tokens['id'] == 17]


# One of the examples is on the train set while the other is on the test set.

# ### Text

# In[42]:


train_balanced_text, test_balanced_text = util.dataset_split(balanced_text, percent=20)


# In[43]:


train_imbalanced_text, test_imbalanced_text = util.dataset_split(imbalanced_text, percent=20)


# Confirm that the train and test datasets do not have any different emails:

# In[44]:


(train_balanced_text['id'] != train_balanced_tokens['id']).any() and (train_imbalanced_text['id'] != train_imbalanced_tokens['id']).any() and (test_balanced_text['id'] != test_balanced_tokens['id']).any() and (test_imbalanced_text['id'] != test_imbalanced_tokens['id']).any()


# ### Saving the Results

# #### Tokens

# In[45]:


save_to_csv(train_balanced_tokens, csv_path, 'train_balanced_tokens.csv')
save_to_csv(test_balanced_tokens, csv_path, 'test_balanced_tokens.csv')


# In[46]:


save_to_csv(train_imbalanced_tokens, csv_path, 'train_imbalanced_tokens.csv')
save_to_csv(test_imbalanced_tokens, csv_path, 'test_imbalanced_tokens.csv')


# #### Text

# In[47]:


save_to_csv(train_balanced_text, csv_path, 'train_balanced_text.csv')
save_to_csv(test_balanced_text, csv_path, 'test_balanced_text.csv')


# In[48]:


save_to_csv(train_imbalanced_text, csv_path, 'train_imbalanced_text.csv')
save_to_csv(test_imbalanced_text, csv_path, 'test_imbalanced_text.csv')


# In[48]:




