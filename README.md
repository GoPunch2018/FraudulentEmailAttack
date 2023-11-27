## Data
In order to download the phishing data (from Jose Nazario's website):
```
cd ~/projects/phishing/data/phishing/nazario/
wget -r --no-parent --level=1 --reject "index.html*" --wait=1 https://monkey.org/~jose/phishing/
cp ./monkey.org/~jose/phishing/* ./
```
Make sure the `--level=1` flag is used, otherwise `wget` will follow every
phishing link found in the emails.

For the legitimate Enron data (from the Carnegie Mellon’s School of Computer
Science website):
```
cd ~/projects/phishing/data/legitimate/enron
wget https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
tar -x -f enron_mail_20150507.tar.gz
```

In case this directory structure is not followed, make sure the path/filename
variables in the code are changed as well.

How to get dataset of LingSpam and SpamAssassin are not listed here, please help yourself.

## Requirements
### pip
To install the dependencies needed to run the code:
```
pip install -r requirements.txt
```
### conda
To install the dependencies needed to run the code:
```
conda create --file environment.yml
```
Please note that, the environment's name and the path where to install the environment 
is decided by 'environment.yml'. Change the path first, otherwise the installation will 
fail because the path doesn't exist in your computer.

Also, in order for Language Tool to work, make sure Java is installed on the
machine.


## Project Structure
Each notebook contains a part of the data processing and classification
process, and outputs the results in .csv files, in order to avoid running the
entire code (a process that can be lengthy) every time a change is being done in
a specific part of the process.

The notebooks should be ran in the following order:
- `Import Text Data.ipynb` It reads the raw data files and stores them in
  `pandas.DataFrame`.
- `Text Dataset Cleanup.ipynb` Some rudimentary processing (like removing empty
  rows) and creation of two datasets with 1:1 and 1:10 phishing to legitimate
  ratios.
- `Text Data Preprocessing.ipynb` Conversion of the email strings to lemmatized
  lists of words for vectorization features and preprocessing for style features.
- `Text Feature Extraction and Feature Selection.ipynb` Vectorization of the text
  to create numeric features usig Word2Vec and TF-IDF and selection of top TF-IDF
  features.
- `Text Data Classification (baseline).ipynb` Baseline algorithm training,
  predictions and evaluation metrics using the vectorized text as features.
- `Stylometric Feature Extraction.ipynb` Creation of stylometric features.
- `Style Features Classification (baseline).ipynb` Baseline algorithm training,
  predictions and evaluation metrics using style features. 
- `Style and Content Classification.ipynb` Final predictions and evaluation using
  feature set merging and model stacking to combine the two feature sets

The python files (`raw_utils.py`, `preprocessing.py`, `features.py` and
`machine_learning.py`) contain utility functions that are being used inside the
notebooks.
