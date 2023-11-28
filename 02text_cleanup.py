import os

from raw_utils import save_to_csv
import preprocessing as util

import pandas as pd
import numpy as np

cwd = os.getcwd()
csv_path = os.path.join(cwd, 'data/csv/')
# Generic Spam
ling_spam_csv = 'ling_spam.csv'
enron_csv = 'enron_text_20000.csv'
# Non Targeted phishing
nazario_csv = 'nazario_full.csv'
spam_assassin_csv = 'spam_assassin.csv'
# 1、lingspam
ling_spam_raw = pd.read_csv(os.path.join(csv_path, ling_spam_csv), dtype={'message': 'object'})
ling_spam_raw = ling_spam_raw.dropna()
ling_spam = ling_spam_raw[ling_spam_raw['message'].apply(util.check_empty) == False]
ling_spam = ling_spam[ling_spam.duplicated(keep='first') == False]
ling_spam_only_body = ling_spam[['message']].rename(columns={'message': 'body'})

ling_spam_only_body.to_csv(csv_path + 'ling_spam_only_body.csv', index=False)

# 2、enron
legit_text_small_raw = pd.read_csv(os.path.join(csv_path, enron_csv), index_col=0, dtype={'body': 'object'})
legit_text_small_raw = legit_text_small_raw.dropna()
legit_text_small = legit_text_small_raw[legit_text_small_raw['body'].apply(util.check_empty) == False]
legit_text_small = legit_text_small[legit_text_small.duplicated(keep='first') == False]
# 3、nazario
phishing_text_raw = pd.read_csv(os.path.join(csv_path, nazario_csv), index_col=0, dtype={'body': 'object'})
phishing_text_raw = phishing_text_raw.dropna()
phishing_text = phishing_text_raw[phishing_text_raw['body'].apply(util.check_empty) == False]
phishing_text = phishing_text[phishing_text['body'].str.contains(
    "This text is part of the internal format of your mail folder, and is not\na real message.") == False]
phishing_text = phishing_text[phishing_text.duplicated(keep='first') == False]
# 4、spam assassin
spam_assassin_raw = pd.read_csv(os.path.join(csv_path, spam_assassin_csv), index_col=0, dtype={'body': 'object'})
spam_assassin_raw = spam_assassin_raw.dropna()
spam_assassin = spam_assassin_raw[spam_assassin_raw['body'].apply(util.check_empty) == False]
spam_assassin = spam_assassin[spam_assassin['body'].str.contains(
    "This text is part of the internal format of your mail folder, and is not\na real message.") == False]
spam_assassin = spam_assassin[spam_assassin.duplicated(keep='first') == False]

ling_spam_only_body['class'] = 0
legit_text_small['class'] = 0
phishing_text['class'] = 1
spam_assassin['class'] = 1

generic_spam = pd.concat([ling_spam_only_body, legit_text_small])
generic_spam = generic_spam.sample(frac=1, random_state=1746).reset_index(drop=True)
generic_spam.insert(0, 'id', generic_spam.index)

save_to_csv(generic_spam, csv_path, 'generic_spam.csv')

non_targeted_phishing = pd.concat([phishing_text, spam_assassin])
non_targeted_phishing = non_targeted_phishing.sample(frac=1, random_state=1746).reset_index(drop=True)
non_targeted_phishing.insert(0, 'id', non_targeted_phishing.index)


save_to_csv(non_targeted_phishing, csv_path, 'non_targeted_phishing.csv')
