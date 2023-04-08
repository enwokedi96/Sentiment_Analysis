from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from featureEngineering import X, y
from sklearn.metrics import accuracy_score
import pickle

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)

# save the model to disk
modelName = r'C:/Users/admin/Documents/pythonDevs/Sentiment_Analysus/NaiiveBayes.sav'

pickle.dump(clf, open(modelName, 'wb'))
  
# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)
