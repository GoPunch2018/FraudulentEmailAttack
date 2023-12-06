import os
import pandas as pd

pd.options.display.max_colwidth = 160

import preprocessing as util
from raw_utils import save_to_csv

cwd = os.getcwd()
csv_path = os.path.join(cwd, 'data/csv/')

data_files = ['enron_tokens.csv', 'nazario_tokens.csv']
# data_files = ['generic_spam.csv', 'non_targeted_phishing.csv']

generic_spam = pd.read_csv(os.path.join(csv_path, data_files[0]), index_col=0,
                           dtype={'body': 'object', 'class': 'bool'})
non_targeted = pd.read_csv(os.path.join(csv_path, data_files[1]), index_col=0,
                           dtype={'body': 'object', 'class': 'bool'})

generic_spam['body'] = generic_spam['body'].apply(util.replace_email)
generic_spam['body'] = generic_spam['body'].apply(util.replace_url)
non_targeted['body'] = non_targeted['body'].apply(util.replace_email)
non_targeted['body'] = non_targeted['body'].apply(util.replace_url)
save_to_csv(generic_spam, csv_path, 'generic_spam_replace_email&url.csv')

save_to_csv(non_targeted, csv_path, 'non_targeted_replace_email&url.csv')
generic_spam_tokens = generic_spam.copy()
non_targeted_tokens = non_targeted.copy()
generic_spam_tokens['body'] = generic_spam_tokens['body'].apply(util.sanitize_addresses)
generic_spam_tokens['body'] = generic_spam_tokens['body'].apply(util.sanitize_url)
generic_spam_tokens['body'] = generic_spam_tokens['body'].apply(util.sanitize_whitespace)
generic_spam_tokens['body'] = generic_spam_tokens['body'].apply(util.remove_non_letters_and_extra_spaces)
generic_spam_tokens['body'] = generic_spam_tokens['body'].apply(util.remove_punctuation)
generic_spam_tokens['body'] = generic_spam_tokens['body'].apply(util.tokenize)
generic_spam_tokens['body'] = generic_spam_tokens['body'].apply(util.remove_stopwords)
generic_spam_tokens['body'] = generic_spam_tokens['body'].apply(util.word_stemming)
generic_spam_tokens['body'] = generic_spam_tokens['body'].apply(util.lemmatize)
# generic_spam_tokens['body'] = generic_spam_tokens['body'].apply(lambda x: ' '.join(x))
generic_spam_tokens = generic_spam_tokens[generic_spam_tokens['body'].astype(bool)]

non_targeted_tokens['body'] = non_targeted_tokens['body'].apply(util.sanitize_addresses)
non_targeted_tokens['body'] = non_targeted_tokens['body'].apply(util.sanitize_url)
non_targeted_tokens['body'] = non_targeted_tokens['body'].apply(util.sanitize_whitespace)
non_targeted_tokens['body'] = non_targeted_tokens['body'].apply(util.remove_non_letters_and_extra_spaces)
non_targeted_tokens['body'] = non_targeted_tokens['body'].apply(util.remove_punctuation)
non_targeted_tokens['body'] = non_targeted_tokens['body'].apply(util.tokenize)
non_targeted_tokens['body'] = non_targeted_tokens['body'].apply(util.remove_stopwords)
non_targeted_tokens['body'] = non_targeted_tokens['body'].apply(util.word_stemming)
non_targeted_tokens['body'] = non_targeted_tokens['body'].apply(util.lemmatize)
# non_targeted_tokens['body'] = non_targeted_tokens['body'].apply(lambda x: ' '.join(x))
non_targeted_tokens = non_targeted_tokens[non_targeted_tokens['body'].astype(bool)]

# save_to_csv(generic_spam_tokens, csv_path, 'generic_spam_tokens.csv')
# save_to_csv(non_targeted_tokens, csv_path, 'non_targeted_tokens.csv')

save_to_csv(generic_spam_tokens, csv_path, 'enron_tokens.csv')
save_to_csv(non_targeted_tokens, csv_path, 'nazario_tokens.csv')
