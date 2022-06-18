import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("dataset/sentiment_train.csv", encoding='latin-1')
df_data = df[["SentimentText", "Sentiment"]]

df_x = df_data['SentimentText']
df_y = df_data['Sentiment']

corpus = df_x
cv = CountVectorizer()
X = cv.fit_transform(corpus)

X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)

clf = LogisticRegression(C=0.5, dual=False, max_iter=10000, solver='lbfgs')
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
filename = 'models/finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))


loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(classification_report(y_test, loaded_model.predict(X_test), digits=4))
