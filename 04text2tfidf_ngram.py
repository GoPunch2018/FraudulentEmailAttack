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

# generic_spam_tokens = 'generic_spam_tokens.csv'
# non_targeted_tokens = 'non_targeted_tokens.csv'
generic_spam_tokens = 'enron_tokens.csv'
non_targeted_tokens = 'nazario_tokens.csv'
generic_spam_tokens = pd.read_csv(os.path.join(csv_path, generic_spam_tokens), index_col=0,
                                  converters={'body': literal_eval})
non_targeted_tokens = pd.read_csv(os.path.join(csv_path, non_targeted_tokens), index_col=0,
                                  converters={'body': literal_eval})

tfidf_balanced = util.tfidf_features_unsupervised(generic_spam_tokens['body'], min_df=5, ngram_range=(2, 3),
                                                  max_features=500, topic_number=10)
W_generic_spam = tfidf_balanced['document-topic']
W_generic_spam = pd.DataFrame(W_generic_spam)
W_generic_spam_max = W_generic_spam.apply(lambda x: (x == x.max()), axis=1).astype(int)
H_generic_spam = tfidf_balanced['topic-term']
tfidf_vectorizer_ge = tfidf_balanced['tfidf_vectorizer']
a = util.get_topics(tfidf_vectorizer_ge, H_generic_spam)
with open('data/generic_spam_topic.txt', 'w') as f:
    for sublist in a:
        # 将每个子列表转换为字符串，然后写入文件
        f.write(str(sublist) + '\n')
topic_dimension = util.enron_topic_dimension_mapping()
document_dimension = np.dot(W_generic_spam, topic_dimension)
document_dimension[document_dimension < 0.01] = 0

result_df = pd.DataFrame(document_dimension)
save_to_csv(result_df, csv_path, 'gs_document-dimension.csv')
tfidf_non_targeted = util.tfidf_features_unsupervised(non_targeted_tokens['body'], min_df=5, ngram_range=(2, 3),
                                                      max_features=500, topic_number=15)
W_non_targeted = tfidf_non_targeted['document-topic']
W_non_targeted = pd.DataFrame(W_non_targeted)
W_generic_spam_max = W_generic_spam.apply(lambda x: (x == x.max()), axis=1).astype(int)
H_non_targeted = tfidf_non_targeted['topic-term']
tfidf_vectorizer_non = tfidf_non_targeted['tfidf_vectorizer']
b = util.get_topics(tfidf_vectorizer_non, H_non_targeted)
with open('data/non_targeted_topic.txt', 'w') as f:
    for sublist in b:
        # 将每个子列表转换为字符串，然后写入文件
        f.write(str(sublist) + '\n')
topic_dimension = util.nazario_topic_dimension_mapping()
document_dimension = np.dot(W_non_targeted, topic_dimension)
document_dimension[document_dimension < 0.01] = 0


result_df = pd.DataFrame(document_dimension)
save_to_csv(result_df, csv_path, 'nts_document-dimension.csv')