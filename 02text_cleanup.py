import os

from raw_utils import save_to_csv
import preprocessing as util

import pandas as pd
import numpy as np

cwd = os.getcwd()
csv_path = os.path.join(cwd, 'data/csv/')
# Generic Spam
ling_spam_csv = 'messages.csv'
enron_csv = 'enron_text.csv'
# Non Targeted phishing
nazario_csv = 'nazario_text.csv'
spam_assassin_csv = 'spam_assassin_text.csv'
# 1、lingspam
ling_spam_raw = pd.read_csv(os.path.join(csv_path, ling_spam_csv), dtype={'message': 'object'})
ling_spam_raw = ling_spam_raw[ling_spam_raw['label'] == 1]
ling_spam_raw.drop('subject', axis=1, inplace=True)
ling_spam_raw = ling_spam_raw.dropna()
ling_spam = ling_spam_raw[ling_spam_raw['message'].apply(util.check_empty) == False]
ling_spam = ling_spam[ling_spam.duplicated(keep='first') == False]
ling_spam = ling_spam[['message']].rename(columns={'message': 'body'})

ling_spam.to_csv(csv_path + 'ling_spam_text.csv', index=True)

# 2、enron
enron_text_raw = pd.read_csv(os.path.join(csv_path, enron_csv), index_col=0, dtype={'body': 'object'})
enron_text_raw = enron_text_raw.dropna()
enron_text = enron_text_raw[enron_text_raw['body'].apply(util.check_empty) == False]
enron_text = enron_text[enron_text.duplicated(keep='first') == False]
# 3、nazario
nazario_text_raw = pd.read_csv(os.path.join(csv_path, nazario_csv), index_col=0, dtype={'body': 'object'})
nazario_text_raw = nazario_text_raw.dropna()
nazario_text = nazario_text_raw[nazario_text_raw['body'].apply(util.check_empty) == False]
nazario_text = nazario_text[nazario_text['body'].str.contains(
    "This text is part of the internal format of your mail folder, and is not\na real message.") == False]
nazario_text = nazario_text[nazario_text.duplicated(keep='first') == False]
# 4、spam assassin
spam_assassin_raw = pd.read_csv(os.path.join(csv_path, spam_assassin_csv), index_col=0, dtype={'body': 'object'})
spam_assassin_raw = spam_assassin_raw.dropna()
spam_assassin = spam_assassin_raw[spam_assassin_raw['body'].apply(util.check_empty) == False]
spam_assassin = spam_assassin[spam_assassin['body'].str.contains(
    "This text is part of the internal format of your mail folder, and is not\na real message.") == False]
spam_assassin = spam_assassin[spam_assassin.duplicated(keep='first') == False]

ling_spam['class'] = 0
enron_text['class'] = 0
nazario_text['class'] = 1
spam_assassin['class'] = 1

generic_spam = pd.concat([ling_spam, enron_text])
generic_spam = generic_spam.sample(frac=1, random_state=1746).reset_index(drop=True)
generic_spam.insert(0, 'id', generic_spam.index)
save_to_csv(enron_text, csv_path, 'enron_prepro.csv')
save_to_csv(generic_spam, csv_path, 'generic_spam.csv')

non_targeted_phishing = pd.concat([nazario_text, spam_assassin])
non_targeted_phishing = non_targeted_phishing.sample(frac=1, random_state=1746).reset_index(drop=True)
non_targeted_phishing.insert(0, 'id', non_targeted_phishing.index)

save_to_csv(non_targeted_phishing, csv_path, 'non_targeted_phishing.csv')
save_to_csv(nazario_text, csv_path, 'nazario_prepro.csv')
