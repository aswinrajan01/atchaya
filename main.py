from flask import Flask, render_template, request
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import spacy
import time
from empath import Empath

app = Flask("__init__")

def POS(x_test):
    # nlp = en_core_web_sm.load()
    nlp = spacy.load("en_core_web_sm")
    x = []
    text_new = []
    doc = nlp(x_test)
    for token in doc:
        text_new.append(token.pos_)
    txt = ' '.join(text_new)
    return txt

def semantics(x_test):
    lexicon = Empath()
    categories = []
    a = lexicon.analyze("")
    for key, value in a.items():
        categories.append(key)
    
    semantic = []
    cnt = 0
    d = lexicon.analyze(x_test)
    sem = []
    for key, value in d.items():
        sem.append(value)
    a = []
    for j in range(len(sem)):
        for k in range(int(sem[j])):
            a.append(categories[j])
        b = " ".join(a)
    return b

def dataFetch():
    df = pd.read_pickle('temp/Semantic.pkl')

    y = df.Label
    x_train, x_test, y_train, y_test = train_test_split(df,y, test_size=0.33)

    x_train_text = x_train['Text']
    x_train_text_pos = x_train['Text_pos']
    x_train_semantics = x_train['Semantics']

    return x_train_text,x_train_text_pos,x_train_semantics,y_train

def vectorize(x_train,x_test,textType = 'text'):
    if textType == 'text':
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range = (2,2), max_features = 20000) 
    elif textType == 'pos':
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range = (2,2))
    elif textType == 'semantic':
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range = (1,1))
    
    tfidf_train = tfidf_vectorizer.fit_transform(x_train.astype('str'))
    tfidf_test = tfidf_vectorizer.transform([x_test])
    return tfidf_train,tfidf_test

def combine(tfidf_train,tfidf_test,tfidf1_train,tfidf1_test,tfidf2_train,tfidf2_test):
    diff_n_rows = tfidf_train.shape[0] - tfidf1_train.shape[0]
    Xb_new = sp.vstack((tfidf1_train, sp.csr_matrix((diff_n_rows, tfidf1_train.shape[1])))) 
    c = sp.hstack((tfidf_train, Xb_new))
    diff_n_rows = c.shape[0] - tfidf2_train.shape[0]
    Xb_new = sp.vstack((tfidf2_train, sp.csr_matrix((diff_n_rows, tfidf2_train.shape[1])))) 
    X = sp.hstack((c, Xb_new))

    dif_n_rows = tfidf_test.shape[0] - tfidf1_test.shape[0]
    Xb_ne = sp.vstack((tfidf1_test, sp.csr_matrix((dif_n_rows, tfidf1_test.shape[1])))) 
    d = sp.hstack((tfidf_test, Xb_ne))
    dif_n_rows = d.shape[0] - tfidf2_test.shape[0]
    Xb_ne = sp.vstack((tfidf2_test, sp.csr_matrix((dif_n_rows, tfidf2_test.shape[1])))) 
    Y = sp.hstack((d, Xb_ne))
    return X,Y

@app.route("/")
def index():
    return render_template("index.html", title="Home | FakeNewsDetector", results=False)


@app.route('/validate', methods=['POST'])
def classify():
    start = time.time()
    print('Start : 0')
    # For testing any new article
    x_test = request.form["news"]

    x_train_text,x_train_text_pos,x_train_semantics,y_train = dataFetch()
    print('Data Fetch : ',time.time()-start)
    tfidf1_train , tfidf1_test = vectorize(x_train_text,x_test)
    print('Vectorize 1 : ',time.time()-start)

    pos = POS(x_test)
    tfidf_train ,tfidf_test = vectorize(x_train_text_pos,pos,'pos')
    print('Vectorize 2 : ',time.time()-start)
    
    semantic = semantics(x_test) 
    tfidf2_train ,tfidf2_test = vectorize(x_train_semantics,semantic,'semantic')
    print('Vectorize 3 : ',time.time()-start)

    big_w = 0.35
    synt_w = 0.5
    sem_w = 0.15
    big_w *= 3
    synt_w *= 3
    sem_w *= 3
    tfidf1_train = big_w * tfidf1_train
    tfidf1_test = big_w * tfidf1_test
    tfidf_train = synt_w * tfidf_train
    tfidf_test = synt_w * tfidf_test
    tfidf2_train = sem_w * tfidf2_train
    tfidf2_test = sem_w * tfidf2_test
    print('Weight Alteration : ',time.time()-start)

    X,Y = combine(tfidf_train,tfidf_test,tfidf1_train,tfidf1_test,tfidf2_train,tfidf2_test)
    print('Combining : ',time.time()-start)

    # modelPath = 'temp/bi_pos_sem_gb'
    # modelPath = 'temp/bi_pos_sem_rf'
    # modelPath = 'temp/bi_pos_sem_nb'

    # clf = pickle.load(open(modelPath,'rb'))
    # out = clf.predict(Y)

    clf = MultinomialNB()
    clf.fit(X, y_train)
    print('Training : ',time.time()-start)
    out = clf.predict(Y)
    print(out)
    return render_template("index.html", title="Results | FakeNewsDetector", results=True, result=out[0],time=time.time()-start)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
