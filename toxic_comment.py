import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

from utils import tokenize


def fit_logistic(inp, out):
    out = out.values
    model_reg = LogisticRegression(C=4, max_iter=1000)
    return model_reg.fit(inp, out)


COMMENT = 'comment_text'
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

print("Load data...")
train = pd.read_csv('dataset/toxic_comment_train.csv')
test = pd.read_csv('dataset/toxic_comment_test.csv')

print("Fill empty with unknown...")
train[COMMENT].fillna('unknown', inplace=True)
test[COMMENT].fillna('unknown', inplace=True)

print("Train TFIDF vectorizer...")
tfidfvectorizer = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
                                  min_df=3, max_df=0.9, strip_accents='unicode',
                                  use_idf=1, smooth_idf=True, sublinear_tf=1)

train_term_doc = tfidfvectorizer.fit_transform(train[COMMENT])
x = train_term_doc

with open('models/tfidf_vectorizer_train.pkl', 'wb') as tfidf_file:
    pickle.dump(tfidfvectorizer, tfidf_file)

print("Fit logistic regression for each class...")
for i, j in enumerate(label_cols):
    print("Fitting:", j)
    model = fit_logistic(x, train[j])

    with open('models/logistic_{}.pkl'.format(j), 'wb') as lg_file:
        pickle.dump(model, lg_file)
