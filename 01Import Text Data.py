#!/usr/bin/env python
# coding: utf-8

# In[1]:


import raw_utils as util
import os

import pandas as pd
import numpy as np

import random
random.seed(1746)


# ## Phishing

# ### Nazario Phishing Corpus

# We will start with reading the subset of the Phishing Corpus that we want.

# In[2]:


# Paths
# cwd = os.getcwd()
cwd = r"D:\01WorkingDirectory"
nazario_path = os.path.join(cwd, 'data/raw/phishing/nazario/')
enron_path = os.path.join(cwd, 'data/enron_mail_20150507/')

csv_path = os.path.join(cwd, 'data/csv/')


# In[3]:


# Files to be ignored for read_dataset()
files_ignored = ['README.txt']
files_ignored_recent = ['README.txt', '20051114.mbox',  'phishing0.mbox',  'phishing1.mbox',  'phishing2.mbox',  'phishing3.mbox', 'private-phishing4.mbox']


# First, we will read and convert all of the dataset. It is straightforward since it is a collection of .mbox files

# In[ ]:


phishing = util.read_dataset(nazario_path, files_ignored, text_only=True)


# In[11]:


phishing.info()


# In[13]:


util.save_to_csv(phishing, csv_path, 'nazario_full.csv')


# Then, we will also take the subset of only the recent emails.

# In[14]:


phishing_recent = util.read_dataset(nazario_path, files_ignored_recent, text_only=True)


# In[15]:


phishing_recent.info()


# In[16]:


util.save_to_csv(phishing_recent, csv_path, 'nazario_recent.csv')


# ## Legitimate

# ### Enron Email Dataset

# This dataset is very big in size so we will just sample different sized sets of random emails from it.

# In[17]:


filename = util.sample_enron_to_mbox(enron_path, 2000)
enron_2000 = util.mbox_to_df(filename, enron_path+'/mbox', text_only=True)
util.save_to_csv(enron_2000, csv_path, 'enron_text_2000.csv')


# In[18]:


filename = util.sample_enron_to_mbox(enron_path, 20000)
enron_20000 = util.mbox_to_df(filename, enron_path+'/mbox', text_only=True)
util.save_to_csv(enron_20000, csv_path, 'enron_text_20000.csv')


# In[ ]:




