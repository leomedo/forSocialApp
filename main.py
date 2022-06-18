import json

from flask import Flask, request
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

with open('models/tfidf_vectorizer_train.pkl', 'rb') as tfidf_file:
    tfidf_model = pickle.load(tfidf_file)
with open('models/logistic_toxic.pkl', 'rb') as logistic_toxic_file:
    logistic_toxic_model = pickle.load(logistic_toxic_file)
with open('models/logistic_severe_toxic.pkl', 'rb') as logistic_severe_toxic_file:
    logistic_severe_toxic_model = pickle.load(logistic_severe_toxic_file)
with open('models/logistic_identity_hate.pkl', 'rb') as logistic_identity_hate_file:
    logistic_identity_hate_model = pickle.load(logistic_identity_hate_file)
with open('models/logistic_insult.pkl', 'rb') as logistic_insult_file:
    logistic_insult_model = pickle.load(logistic_insult_file)
with open('models/logistic_obscene.pkl', 'rb') as logistic_obscene_file:
    logistic_obscene_model = pickle.load(logistic_obscene_file)
with open('models/logistic_threat.pkl', 'rb') as logistic_threat_file:
    logistic_threat_model = pickle.load(logistic_threat_file)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route("/comment_toxic")
def comment_toxic():
    comment = request.args.get('comment')
    comment_term_doc = tfidf_model.transform([comment])
    dict_preds = {'toxic': logistic_toxic_model.predict_proba(comment_term_doc)[:, 1][0],
                  'severe_toxic': logistic_severe_toxic_model.predict_proba(comment_term_doc)[:, 1][0],
                  'identity_hate': logistic_identity_hate_model.predict_proba(comment_term_doc)[:, 1][0],
                  'insult': logistic_insult_model.predict_proba(comment_term_doc)[:, 1][0],
                  'obscene': logistic_obscene_model.predict_proba(comment_term_doc)[:, 1][0],
                  'threat': logistic_threat_model.predict_proba(comment_term_doc)[:, 1][0]}

    for k in dict_preds:
        perc = dict_preds[k] * 100
        dict_preds[k] = "{0:.2f}%".format(perc)
    return json.dumps(dict_preds, ensure_ascii=False)


@app.route('/sentiment_train')
def sentiment_train():
    post = request.args.get('post')
    df = pd.read_csv("dataset/sentiment_train.csv", encoding='latin-1')
    df_data = df[["SentimentText", "Sentiment"]]
    df_x = df_data['SentimentText']
    corpus = df_x
    cv = CountVectorizer()
    cv.fit_transform(corpus)
    filename = 'models/finalized_model.sav'
    clf = pickle.load(open(filename, 'rb'))
    data = [post]
    vect = cv.transform(data).toarray()
    my_prediction = clf.predict(vect)[0]
    if my_prediction == 1:
        result = 'Positive'
    else:
        result = 'Negative'
    return json.dumps({'result': result}, ensure_ascii=False)


@app.route('/lang_detection')
def lang_detection():
    text = request.args.get('text')
    df = pd.read_csv("dataset/Language Detection.csv")
    X = df["Text"]
    y = df["Language"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    cv = CountVectorizer()
    X = cv.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=32)
    model = MultinomialNB()
    model.fit(x_train, y_train)

    data = [text]
    vect = cv.transform(data).toarray()
    my_prediction = model.predict(vect)
    corr_language = le.inverse_transform(my_prediction)
    output = corr_language[0]

    return json.dumps({'result': output}, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
