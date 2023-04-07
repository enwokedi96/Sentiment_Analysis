
'''
DATASET:: https://ai.stanford.edu/~amaas/data/sentiment/
Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and 
Christopher Potts. (2011). 
Learning Word Vectors for Sentiment Analysis. 
The 49th Annual Meeting of the Association for 
Computational Linguistics (ACL 2011).
'''

import os
import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.utils import shuffle

# nltk.download('punkt')
# nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    return ' '.join(tokens)

def load_data(path):
    df = pd.DataFrame()
    for label in ['pos', 'neg']:
        folder = os.path.join(path, label)
        files = os.listdir(folder)
        texts = []
        for file in files:
            with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
                text = f.read()
                text = preprocess_text(text)
                texts.append(text)
        labels = [1 if label == 'pos' else 0] * len(files)
        df_temp = pd.DataFrame({'text': texts, 'label': labels})
        df = pd.concat([df, df_temp], axis=0)
    df = shuffle(df)
    df = df.reset_index(drop=True)
    return df
