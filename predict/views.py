from django.shortcuts import render
from django.http import HttpResponse
import joblib
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

def cleaningUlasan(text):
    text = re.sub(r'[0-9]+',' ',text)
    text = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", text)
    text = text.strip(' ')
    return text

def caseFolding(text):
    text = text.lower()
    return text

def tokenizingText(text):
    text = word_tokenize(text)
    return text

stop_words = set(stopwords.words('indonesian'))
def stopwordRemoval(text):
    return [word for word in text if word not in stop_words]

def joinWord(text):
    return " ".join(text)

factory = StemmerFactory()
stemmer = factory.create_stemmer()
def stemming(text):
    text = [stemmer.stem(word) for word in text]
    return text

def convertToLabel(input):
    if input == 0:
        return "negative"
    elif input == 2:
        return "positive"
    else:
        return "neutral"



def index(request):
    return render(request, "predict/index.html")

def predict(request):
    SVM = joblib.load('SVM.joblib')
    KNN = joblib.load('KNN.joblib')
    LR = joblib.load('LR.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
        


    if request.method =='POST':
        input = request.POST.get('review')

        input = tokenizingText(input)
        input = stopwordRemoval(input)
        input = stemming(input)
        input = joinWord(input)
        input = [input]

        input = vectorizer.transform(input)

        return render(request, "predict/hasil.html",{
            'SVM':convertToLabel(SVM.predict(input)),
            'KNN':convertToLabel(KNN.predict(input)),
            'LR':convertToLabel(LR.predict(input))
        })







