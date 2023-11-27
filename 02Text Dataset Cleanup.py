#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

from raw_utils import save_to_csv
import preprocessing as util

import pandas as pd
import numpy as np


# In[2]:


# Paths
cwd = os.getcwd()
csv_path = os.path.join(cwd, 'data/csv/')

# Filenames
nazario_csv = 'nazario_recent.csv'
enron_csv = ['enron_text_2000.csv', 'enron_text_20000.csv']


# ## Phishing

# First, read the csv with the recent emails.

# In[3]:


phishing_text_raw = pd.read_csv(os.path.join(csv_path, nazario_csv), index_col=0, dtype={'body': 'object'})


# In[4]:


phishing_text_raw.info()


# ### Cleanup

# #### Remove Uninformative Rows

# There are some rows with `null` bodies. Those need to be dropped in order for the other functions to work.

# In[5]:


phishing_text_raw = phishing_text_raw.dropna()


# There are multipart emails that were empty except for attachments, so they can also be dropped.

# In[6]:


phishing_text = phishing_text_raw[phishing_text_raw['body'].apply(util.check_empty) == False]
a = phishing_text_raw['body'].apply(util.check_empty) == False
phishing_text.shape


# Afterwards, we can see that there are some computer generated messages at the beginning of the mbox files, which we also need to remove.

# In[7]:


phishing_text = phishing_text[phishing_text['body'].str.contains("This text is part of the internal format of your mail folder, and is not\na real message.") == False]
phishing_text.shape


# Finally, the duplicate rows will be removed.

# In[8]:


phishing_text = phishing_text[phishing_text.duplicated(keep='first') == False]
phishing_text.shape


# There were a lot of duplicates and now 1688 emails remain to work with.

# ## Legitimate

# This process will be repeated with the two legitimate email datasets (since we aim for ratios of 1:1 and 1:10).

# In[9]:


legit_text_small_raw = pd.read_csv(os.path.join(csv_path, enron_csv[0]), index_col=0, dtype={'body': 'object'})
legit_text_small_raw.info()


# In[10]:


legit_text_big_raw = pd.read_csv(os.path.join(csv_path, enron_csv[1]), index_col=0, dtype={'body': 'object'})
legit_text_big_raw.info()


# ### Cleanup

# #### Remove Uninformative Rows

# Drop `null` rows.

# In[11]:


legit_text_small_raw = legit_text_small_raw.dropna()
legit_text_small_raw.shape


# In[12]:


legit_text_big_raw = legit_text_big_raw.dropna()
legit_text_big_raw.shape


# Check for empty emails.

# In[13]:


legit_text_small = legit_text_small_raw[legit_text_small_raw['body'].apply(util.check_empty) == False]
legit_text_small.shape


# In[14]:


legit_text_big = legit_text_big_raw[legit_text_big_raw['body'].apply(util.check_empty) == False]
legit_text_big.shape


# There are no computer generated emails like those removed above in this dataset, so only the duplicates need removal.

# In[15]:


legit_text_small = legit_text_small[legit_text_small.duplicated(keep='first') == False]
legit_text_small.shape


# In[16]:


legit_text_big = legit_text_big[legit_text_big.duplicated(keep='first') == False]
legit_text_big.shape


# ## Mixed Datasets

# Finally, the two mixed datasets will be created, adding an extra column that shows the class (phishing or legitimate).<br>
# In addition, the datasets will be shuffled and a column containing a unique identifier will be added.

# In[17]:


phishing_text['class'] = 1


# #### 1:1 ratio

# In[18]:


legit_text_small = legit_text_small.sample(n=1688, random_state=1746)
legit_text_small['class'] = 0


# In[19]:


balanced = pd.concat([phishing_text, legit_text_small])
balanced = balanced.sample(frac=1, random_state=1746).reset_index(drop=True)
balanced.insert(0, 'id', balanced.index)


# In[20]:


save_to_csv(balanced, csv_path, 'balanced.csv')


# #### 1:10 ratio

# In[21]:


legit_text_big = legit_text_big.sample(n=16880, random_state=1746)
legit_text_big['class'] = 0


# In[22]:


imbalanced = pd.concat([phishing_text, legit_text_big])
imbalanced = imbalanced.sample(frac=1, random_state=1746).reset_index(drop=True)
imbalanced.insert(0, 'id', imbalanced.index)


# In[23]:


save_to_csv(imbalanced, csv_path, 'imbalanced.csv')


# In[23]:




