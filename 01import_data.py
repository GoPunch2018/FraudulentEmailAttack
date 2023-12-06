import raw_utils as util
import os

import pandas as pd
import numpy as np

import random

random.seed(1746)

cwd = r"D:\01WorkingDirectory"
# Generic Spam
ling_spam_path = os.path.join(cwd, 'data/LingSpam')
enron_path = os.path.join(cwd, 'data/enron_mail_20150507/')
# Non Targeted phishing
nazario_path = os.path.join(cwd, 'data/nazario/')
spam_assassin_path = os.path.join(cwd, 'data/SpamAssassin/spam_2')

csv_path = os.path.join(cwd, 'data/csv/')
# 1、处理lingspam数据，ling_spam已经拥有了index,message,label
# 的格式，我们读取label为1的message（钓鱼邮件）就可以
# 2、处理enron数据集，该数据集拥有46000多封邮件，数据量太大，暂时抽取2000或者20000封，形成我们的数据集
# 最终使用enron_text_20000.csv
filename = util.sample_enron_to_mbox(enron_path, 46502)
enron_ = util.mbox_to_df(filename, enron_path + '/mbox', text_only=True)
util.save_to_csv(enron_, csv_path, 'enron_text.csv')
# 3、处理nazario数据集
files_ignored = ['README.txt', 'private-phishing4.mbox']
files_ignored_recent = ['README.txt', '20051114.mbox', 'phishing0.mbox', 'phishing1.mbox', 'phishing2.mbox',
                        'phishing3.mbox', 'private-phishing4.mbox']
phishing = util.read_dataset(nazario_path, files_ignored, text_only=True)
util.save_to_csv(phishing, csv_path, 'nazario_text.csv')

# 4、处理spam_assassin数据集
filename = util.spam_assassin_to_mbox(spam_assassin_path)
phishing_spam_assassin = util.mbox_to_df(filename, spam_assassin_path + '/mbox', text_only=True)
util.save_to_csv(phishing_spam_assassin, csv_path, 'spam_assassin_text.csv')
