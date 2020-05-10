from django.http import HttpResponse
from django.shortcuts import render
from django.db import models
from .forms import SearchForm
from .forms import SearchForm2
from dateutil import parser

import timeit
import datetime
import os
import nltk
import re
import pandas as pd
import string
import ast
import xml.dom.minidom
import math
from itertools import islice

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


#Language Model

def join_tags(tokens):
    for i in enumerate(tokens):
        hit = 0
        index = i[0]
        y = i[1]
        width = len(y)
        
        for x in y:
            if x is '-':
                tokens[index] = tokens[index][0:hit] + ' ' + tokens[index][hit+1:]
            hit += 1
        
        if tokens[index] is ' ':
            tokens[index] = ''
        
        if tokens[index] == '  ':
            tokens[index] = ''
        
        if y[(width-1):width] is '.':
            tokens[index] = y[0:(width-1)]
        
        if tokens[index][0:1] is '-' or tokens[index][1:2] is '-':
            tokens[index] = ''
        
        if tokens[index][-2:-1] is "'":
            tokens[index] = tokens[index][0:-2]
        
        if  tokens[index][:1] is "'" :
            tokens[index] = tokens[index][1:]
        
        if tokens[index] is "s" or tokens[index] is "re":
            tokens[index-1] = tokens[index-1]
            tokens[index] = ''
        
        if tokens[index] is '<' and tokens[index+2] is '>':
            tokens[index+2] = '<'+ tokens[index+1]+'>'
            tokens[index] =''
            tokens[index+1] = ''
            
        if tokens[index] is '&' and tokens[index+2] is ';':
            tokens[index+2] = ''
            tokens[index] = ''
            tokens[index+1] = ''
            
        if tokens[index] is '(' or tokens[index] is ')':
            tokens[index] = ''
            
        if tokens[index][0:1] is ':':
            tokens[index] = ''
        
        if tokens[index][0:1] is '.' and tokens[index][1:2] is '.':
            tokens[index] = ''

        if y[(width-2):(width-1)] == '.' and y[(width-1):(width)] == '0':
            tokens[index] = y[0:(width-2)]
        
    tokens_clean = [x for x in tokens if x != '' and x not in ".,!?'" and x not in '<>']
        
    return tokens_clean

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def guess_date(string):
    for fmt in ["%Y/%m/%d", "%d-%m-%Y", "%Y%m%d", "%d-%b-%y", "%d-%m-%y", "%d-%b-%Y", "%d %b %y", "%d %m %y", "%d %b %Y"]:
        try:
            return str(datetime.datetime.strptime(string, fmt).date())
        except ValueError:
            continue
    return string

def preprocess(text_process):
    text_process = text_process.lower()
    tokens = word_tokenize(text_process)
    tokens_clean = join_tags(tokens)
    tokens_clean = [guess_date(i) for i in tokens_clean]
    stop_words = set(stopwords.words('english'))
    filtered_words = [guess_date(w) for w in tokens_clean if not w in stop_words]
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in stemmed_words]
    stemmed_text = ' '.join(stemmed_words)
    lemmatized_text = ' '.join(lemmatized_words)
    clean_text = lemmatized_text
    clean_words = sorted(list([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in stemmed_words]))
    return clean_text, clean_words

def build_index_language_model():
    dict_term_doc = dict()
    clean_words_list = list()
    raw_words_list = list()
    dict_term_doc_final = dict()
    data = dict()
    key = list()
    value = list()
    doc_freqs = dict()
    
    start = timeit.default_timer()
    for index, i in enumerate(os.listdir('Data')):
        try:
            filename = i
            f = open('Data/' + filename, 'r')
            text = f.read()
            f.close()

            clean_text, clean_words = preprocess(text)
            clean_words_list.extend(clean_words)

            doc_freq = 0
            for w in clean_words:
                doc_freq+=1
                doc_freqs[filename[4:7]] = doc_freq
                if w not in dict_term_doc:
                    dict_term_doc[w] = [1, {filename[4:7]: 1}]
                else:
                    if filename[4:7] in dict_term_doc[w][1]:
                        dict_term_doc[w][1][filename[4:7]]+=1
                    else:
                        dict_term_doc[w][1][filename[4:7]]=1
                    dict_term_doc[w][0]+=1
        except Exception as e:
            print(str(e))
    
    stop = timeit.default_timer()
    return stop-start

def language_model(query, limit, lambd):
    clean_query_text, clean_query_words = preprocess(query)
    result_dict = list()

    df_doc_freq = pd.read_csv('Document Frequencies.csv')
    df = pd.read_csv('Frequencies.csv')

    global_freq = int(open("global_frequency.txt", "r").read())

    start = timeit.default_timer()
    for i in df_doc_freq.iterrows():
        local_freq = i[1]['Frequency']
        doc_number = i[1]['Document']
        doc_number = str(doc_number).zfill(3)
        result = 1
        for j in clean_query_words:
            entries = df.loc[df['Term'] == j]['Documents'].values[0]
            entries = entries.split(', ',1)
            entries[0] = entries[0].replace("[", "")
            entries[1] = entries[1].replace(']', '')
            entries[1] = ast.literal_eval(entries[1])
            if doc_number in entries[1].keys():
                prob = (float(entries[1][doc_number])/float(local_freq)*lambd) + (float(entries[0])/float(global_freq)*(1-lambd))
            else:
                prob = (float(entries[0])/float(global_freq)*(1-lambd))
            result *= prob
        result_dict.append((doc_number, result))
    result_sorted = sorted(result_dict, key=lambda x: x[1], reverse=True)
    stop = timeit.default_timer()
    return result_sorted[:limit], stop-start
    
def home(request):
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid() and len(form.cleaned_data.get('query')) != 0:
            query = form.cleaned_data.get('query')
            limit = form.cleaned_data.get('limit')
            lambdaa = form.cleaned_data.get('lambdaa')
            user_input = query.lower()
            

            docs_results = language_model(user_input, limit, lambdaa)
            founded = len(docs_results[0])
            times = docs_results[1]
            
            documents = docs_results[0]

            results = []
            # read xml
            for key,value in documents:
                dictionary = dict()
                filename = 'Doc0' + key + '.xml'
                print('filename: ', filename)
                doc = xml.dom.minidom.parse('C:/Users/Asus/Downloads/Git/search_engine/XML/' + filename)
                title = doc.getElementsByTagName('TITLE')
                title = title[0].firstChild.nodeValue

                date = doc.getElementsByTagName('DATE')
                date = date[0].firstChild.nodeValue
                body = doc.getElementsByTagName('BODY')
                body = body[0].firstChild.nodeValue

                dictionary['document_no'] = key
                dictionary['score'] = value
                dictionary['title'] = title
                dictionary['date'] = date
                dictionary['body'] = body
                results.append(dictionary)

            return render(request, 'Index.html', {'results' : results, 'times': times, 'founded': founded,  'query': query, 'limit' : len(results), 'lambdaa' : lambdaa, })
    else:
        form = SearchForm()
    return render(request, 'Index.html')


# TF IDF
def build_index_tf_idf():
    dict_term_doc = dict()
    clean_words_list = list()
    raw_words_list = list()
    dict_term_doc_final = dict()
    data = dict()
    key = list()
    value = list()
    
    start = timeit.default_timer()
    for i in os.listdir('Data'):
        try:
            filename = i
            f = open('Data/' + filename, 'r')
            text = f.read()
            f.close()

            clean_text, clean_words = preprocess(text)
            clean_words_list.extend(clean_words)
            for w in clean_words:
                if w not in dict_term_doc:
                    dict_term_doc[w] = [filename[4:7]]
                else:
                    dict_term_doc[w].append(filename[4:7])
        except Exception as e:
            print(str(e))

    stop = timeit.default_timer()
    
    return stop-start

def tfidf(query):
    df_tfidf = pd.read_csv('Konstruksi Indeks.csv')
    doc = df_tfidf['Documents']
    term = df_tfidf['Term']

    text, data = preprocess(query)
    z=1
    count = 1
    N = 500+1
    max_range = len(term)

    dict_w = dict()
    dict_f = dict()
    dict_tf = dict()
    dict_idf = dict()
    dict_dj = dict()
    dict_sim = dict()

    starts = timeit.default_timer()
    for q in data:
        for i in range (0,max_range):
            ni = 0
            start = 2
            end = 5
            count = 1
            if (term[i] == q):
                for y in range (0,int(len(doc[i])/7)):
                    if(int(len(doc[i])/7)!=1):
                        if(doc[i][start:end] != doc[i][(start-7):(end-7)]):
                            ni+=1
                            dict_f['f',doc[i][start:end]]=count
                        else:
                            dict_f['f',doc[i][start:end]]=count
                            count+=1
                            ni=1
                    else:
                        dict_f['f',doc[i][start:end]]=count
                        ni+=1

                    start = start+7
                    end = end+7

                dict_idf[q] = math.log((1+N/ni),10)


    for key in dict_f:
        dict_tf['tf',z] = 1+math.log(dict_f[key],10)
        z+=1

    z=1

    for key in dict_tf:
        for key2 in dict_idf:
            dict_w['w',z] = dict_tf[key]*dict_idf[key2]
            z+=1

    z=1

    for key in dict_w:
        dict_dj['dj',z] = math.sqrt(pow(dict_w[key],2))
        z+=1

    z=1
    for key in dict_dj:
        for key2 in dict_w:
            dict_sim["d",z] = dict_w[key2]/dict_dj[key]
            z+=1
            break
    stop = timeit.default_timer()
    result = [(i[1], j)for i,j in dict_f.items()]
    
    return result, stop-starts 
    
def toprank(request):
    if request.method == 'POST':
        form = SearchForm2(request.POST)
        if form.is_valid() and len(form.cleaned_data.get('query')) != 0:
            query = form.cleaned_data.get('query')
            user_input = query.lower()
            docs_results = tfidf(user_input)
            
            times = docs_results[1]
            
            documents = docs_results[0]
            founded = len(documents)

            result_dict = list()
            results = []

            for key,value in documents:
                result_dict.append((key, value))
                result_sorted = sorted(result_dict, key=lambda x: x[1], reverse=True)
            
            for key, value in result_sorted:
                dictionary = dict()
                filename = 'Doc0' + key + '.xml'
                print('filename: ', filename)
                doc = xml.dom.minidom.parse('C:/Users/Asus/Downloads/Git/search_engine/XML/' + filename)
                title = doc.getElementsByTagName('TITLE')
                title = title[0].firstChild.nodeValue

                date = doc.getElementsByTagName('DATE')
                date = date[0].firstChild.nodeValue
                body = doc.getElementsByTagName('BODY')
                body = body[0].firstChild.nodeValue
                dictionary['document_no'] = key
                dictionary['score'] = value
                dictionary['title'] = title
                dictionary['date'] = date
                dictionary['body'] = body
                results.append(dictionary)
            
            return render(request, 'TopRank.html', {'results' : results, 'times': times, 'founded': founded, 'query': query, })
        
    else:
        form = SearchForm2()
    return render(request, 'TopRank.html')

# def buttontoprank(request):
#     return render(request, 'TopRank.html', {'show': show})
