from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import load_data 
import tarfile

# open file 
# file = tarfile.open('C:/Users/admin/Documents/pythonDevs/aclImdb_v1.tar.gz')
# file.extractall('./aclImdb_v1')
# file.close()

# vectorize 
vectorizer = CountVectorizer()

# load data and vectorize
df = load_data(path=r"C:/Users/admin/Documents/pythonDevs/Sentiment_Analysus/aclImdb/train/")
X = vectorizer.fit_transform(df['text'])
y = df['label']
