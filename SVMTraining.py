from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from featureEngineering import X, y
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for vectorization and classification
# pipeline = Pipeline([
#     ('vect', TfidfVectorizer()),
#     ('clf', LinearSVC())
# ])
clf = LinearSVC()

# Train the model
clf.fit(X_train, y_train)

# save the model to disk
modelName = r'C:/Users/admin/Documents/pythonDevs/Sentiment_Analysus/SVM.sav'
pickle.dump(clf, open(modelName, 'wb'))

# Evaluate the model on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)
