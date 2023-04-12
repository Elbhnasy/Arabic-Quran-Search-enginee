import pandas as pd
import re
import string
import joblib

import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
import pyarabic.araby as araby
from nltk.corpus import stopwords
from arabicstopwords.arabicstopwords import stopwords_list
from nltk.stem.snowball import ArabicStemmer
from qalsadi.lemmatizer import Lemmatizer
from flask import Flask, render_template, request


# # load the saved model from file using joblib
# vectorizer = joblib.load('DeployingData/model.pkl')

# Load necessary data and models
df = pd.read_csv(r"DeployingData/processed_df.csv")

st = ArabicStemmer()
lemmer = Lemmatizer()

def normalize_chars(txt):
    txt = re.sub("[إأٱآا]", "ا", txt)
    txt = re.sub("ى", "ي", txt)
    txt = re.sub("ة", "ه", txt)
    return txt



def clean_txt(txt, stopwordlist, lemmer):
    # remove tashkeel & tatweel
    txt = araby.strip_diacritics(txt)
    txt = araby.strip_tatweel(txt)
    # normalize chars
    txt = normalize_chars(txt)
    # remove stopwords & punctuation
    txt = ' '.join([token.translate(str.maketrans('','',string.punctuation)) for token in txt.split(' ') if token not in stopwordlist])
    # lemmatizer
    txt_lemmatized = ' '.join([lemmer.lemmatize(token) for token in txt.split(' ')])
    return txt + " " + txt_lemmatized



def show_best_results(df_quran, scores_array, top_n=20):
    results = []
    sorted_indices = scores_array.argsort()[::-1]
   
    for position, idx in enumerate(sorted_indices[:top_n]):
        row = df_quran.iloc[idx]
        ayah = row["ayah_txt"]
        ayah_num = row["ayah_num"]
        surah_name = row["surah_name"]
        score = scores_array[idx]
        if score > 0:
            result_dict = {
                "ayah": ayah,
                "ayah_num": ayah_num,
                "surah_name": surah_name
            }
            results.append(result_dict)
    return results
# Extract the clean text from the dataframe
corpus = df['clean_txt']

# Instantiate the vectorizer object
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

# Vectorize the corpus
corpus_vectorized = vectorizer.fit_transform(corpus)
def run_arabic_search_engine(query):

    stopwordlist = set(list(stopwords_list()) + stopwords.words('arabic'))
    stopwordlist = [normalize_chars(word) for word in stopwordlist]
    st = ArabicStemmer()
    lemmer = Lemmatizer()

    # Preprocess the query
    query = clean_txt(query, stopwordlist, lemmer)

    # Run the search engine
    corpus = df["clean_txt"]
    # Instantiate the vectorizer object
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    corpus_vectorized = vectorizer.fit_transform(corpus)
    query_vectorized = vectorizer.transform([query])
    scores = query_vectorized.dot(corpus_vectorized.transpose())
    scores_array = scores.toarray()[0]

    # Return the results as a list of dictionaries
    return show_best_results(df, scores_array)



app = Flask(__name__)

@app.route('/')
def search():
    return render_template('search.html')

@app.route('/results', methods=['POST'])
def results():
    query = request.form['query']
    results = run_arabic_search_engine(query)
    return render_template('results.html', query=query, results=results)
if __name__ == '__main__':
    app.run(debug=True)
