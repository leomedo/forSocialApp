import pickle
import pandas as pd
import re
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

warnings.simplefilter("ignore")

df = pd.read_csv("Language Detection.csv")

X = df["Text"]
y = df["Language"]

le = LabelEncoder()
y = le.fit_transform(y)

text_list = []

for text in X:
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    text_list.append(text)

cv = CountVectorizer()
X = cv.fit_transform(text_list).toarray()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=32)

model = MultinomialNB()
model.fit(x_train, y_train)
pickle.dump(model, open("model.pkl", "wb"))
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy is :", accuracy)


def predict(text_lang):
    x = cv.transform([text_lang]).toarray()
    language = model.predict(x)
    lang = le.inverse_transform(language)
    print("The language is in", lang[0])


predict("hombre")
