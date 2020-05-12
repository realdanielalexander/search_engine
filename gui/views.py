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

import json
from django.http import Http404, HttpResponse

# Language Model


def join_tags(tokens):
    for i in enumerate(tokens):
        hit = 0
        index = i[0]
        y = i[1]
        width = len(y)

        for x in y:
            if x is '-':
                tokens[index] = tokens[index][0:hit] + \
                    ' ' + tokens[index][hit+1:]
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

        if tokens[index][:1] is "'":
            tokens[index] = tokens[index][1:]

        if tokens[index] is "s" or tokens[index] is "re":
            tokens[index-1] = tokens[index-1]
            tokens[index] = ''

        if tokens[index] is '<' and tokens[index+2] is '>':
            tokens[index+2] = '<' + tokens[index+1]+'>'
            tokens[index] = ''
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

    tokens_clean = [x for x in tokens if x !=
                    '' and x not in ".,!?'" and x not in '<>']

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
    filtered_words = [guess_date(w)
                      for w in tokens_clean if not w in stop_words]
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(
        w, get_wordnet_pos(w)) for w in stemmed_words]
    stemmed_text = ' '.join(stemmed_words)
    lemmatized_text = ' '.join(lemmatized_words)
    clean_text = lemmatized_text
    clean_words = sorted(
        list([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in stemmed_words]))
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
                doc_freq += 1
                doc_freqs[filename[4:7]] = doc_freq
                if w not in dict_term_doc:
                    dict_term_doc[w] = [1, {filename[4:7]: 1}]
                else:
                    if filename[4:7] in dict_term_doc[w][1]:
                        dict_term_doc[w][1][filename[4:7]] += 1
                    else:
                        dict_term_doc[w][1][filename[4:7]] = 1
                    dict_term_doc[w][0] += 1
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
        else_terus = True
        for j in clean_query_words:
            entries = df.loc[df['Term'] == j]['Documents'].values[0]
            entries = entries.split(', ', 1)
            entries[0] = entries[0].replace("[", "")
            entries[1] = entries[1].replace(']', '')
            entries[1] = ast.literal_eval(entries[1])
            if doc_number in entries[1].keys():
                prob = (float(entries[1][doc_number])/float(local_freq) *
                        lambd) + (float(entries[0])/float(global_freq)*(1-lambd))
                else_terus = False
            else:
                prob = (float(entries[0])/float(global_freq)*(1-lambd))
            result *= prob
        if(else_terus):
            continue
        # if len(result_dict) > 0 and result_dict[-1][1]-result > 1:
        #     break
        result_dict.append((doc_number, result))
    result_sorted = sorted(result_dict, key=lambda x: x[1], reverse=True)
    stop = timeit.default_timer()
    if limit > len(result_sorted):
        limit = len(result_sorted)
    return result_sorted[:limit], stop-start


def language_model_ascending(query, limit, lambd):
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
        else_terus = True
        for j in clean_query_words:
            entries = df.loc[df['Term'] == j]['Documents'].values[0]
            entries = entries.split(', ', 1)
            entries[0] = entries[0].replace("[", "")
            entries[1] = entries[1].replace(']', '')
            entries[1] = ast.literal_eval(entries[1])
            if doc_number in entries[1].keys():
                prob = (float(entries[1][doc_number])/float(local_freq) *
                        lambd) + (float(entries[0])/float(global_freq)*(1-lambd))
                else_terus = False
            else:
                prob = (float(entries[0])/float(global_freq)*(1-lambd))
            result *= prob
        if(else_terus):
            continue
        # if len(result_dict) > 0 and result_dict[-1][1]-result > 1:
        #     break
        result_dict.append((doc_number, result))
    result_sorted = sorted(result_dict, key=lambda x: x[0])
    stop = timeit.default_timer()
    if limit > len(result_sorted):
        limit = len(result_sorted)
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

            result_dict = list()
            results = []

            result_sorted = sorted(documents, key=lambda x: x[0], reverse=True)

            results = []
            # read xml
            for key, value in result_sorted:
                dictionary = dict()
                filename = 'Doc0' + key + '.xml'
                doc = xml.dom.minidom.parse('XML/' + filename)
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

            return render(request, 'gui/Index.html', {'results': results, 'times': times, 'founded': founded,  'query': query, 'limit': len(results), 'lambdaa': lambdaa, })
    else:
        form = SearchForm()
    return render(request, 'gui/Index.html')


def home_ascending(request):
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid() and len(form.cleaned_data.get('query')) != 0:
            query = form.cleaned_data.get('query')
            limit = form.cleaned_data.get('limit')
            lambdaa = form.cleaned_data.get('lambdaa')
            user_input = query.lower()

            docs_results = language_model_ascending(user_input, limit, lambdaa)
            founded = len(docs_results[0])
            times = docs_results[1]

            documents = docs_results[0]

            results = []
            # read xml
            for key, value in documents:
                dictionary = dict()
                filename = 'Doc0' + key + '.xml'
                doc = xml.dom.minidom.parse('XML/' + filename)
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

            return render(request, 'gui/IndexAscending.html', {'results': results, 'times': times, 'founded': founded,  'query': query, 'limit': len(results), 'lambdaa': lambdaa, })
    else:
        form = SearchForm()
    return render(request, 'gui/IndexAscending.html')


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


def tfidf(kueri, limit):
    df_tfidf = pd.read_csv('Konstruksi Indeks.csv')
    doc = df_tfidf['Documents']
    term = df_tfidf['Term']
    df_tfidf
    max_range = len(term)

    query = kueri
    text, data = preprocess(query)

    z = 1
    count = 1
    N = 500
    plus = 0
    list_angka = list()
    list_nama = list()
    list_final = list()
    list_finalnama = list()

    list_wnama = list()
    list_wangka = list()
    list_simnama = list()
    list_simdoc = list()
    list_simq = list()
    list_simqnama = list()
    list_sim = list()
    list_finalsim = list()
    list_finalsim2 = list()
    list_finalsim2doc = list()

    dict_w = dict()
    dict_f = dict()
    dict_tf = dict()
    dict_idf = dict()
    dict_dj = dict()
    dict_dj2 = dict()
    dict_sim = dict()

    list_rank = list()
    counter = 0

    starts = timeit.default_timer()

    for q in data:
        if data[counter] == data[counter-1]:
            plus += 1
            dict_f[q] = plus
        else:
            plus = 1
            dict_f[q] = plus
        counter += 1

    for q in data:
        for i in range(0, max_range):
            ni = dict_f[q]
            start = 2
            end = 5
            count = 1
            if (term[i] == q):
                for y in range(0, int(len(doc[i])/7)):
                    if(int(len(doc[i])/7) != 1):
                        if(doc[i][start:end] != doc[i][(start-7):(end-7)]):
                            ni += 1
                            dict_f[q + ' doc ' + doc[i][start:end]] = count
                        else:
                            dict_f[q + ' doc' + doc[i][start:end]] = count
                            count += 1
                            ni = 1
                    else:
                        dict_f[q + ' doc ' + doc[i][start:end]] = count
                        ni += 1

                    start = start+7
                    end = end+7

                dict_idf[q] = math.log((1+N/ni), 10)

    for key in dict_f:
        dict_tf[key] = 1+math.log(dict_f[key], 10)

    for key in dict_tf:
        for key2 in dict_idf:
            if key[0:len(key)-8] == key2:
                dict_w[key] = dict_tf[key]*dict_idf[key2]
            elif key == key2:
                dict_w[key] = dict_tf[key]*dict_idf[key2]

    for key in dict_w:
        if(key[len(key)-7:len(key)-4] == 'doc'):
            dict_dj[key[len(key)-7:len(key)]] = (pow(dict_w[key], 2))
            list_angka.append(dict_dj[key[len(key)-7:len(key)]])
            list_nama.append(key[len(key)-7:len(key)])
        else:
            dict_dj[key] = pow(dict_w[key], 2)
            list_nama.append('q')
            list_angka.append(dict_dj[key])

    for i in range(0, len(list_nama)):
        for x in range(i+1, len(list_nama)):
            if (list_nama[i] == list_nama[x]):

                list_final.append(math.sqrt(list_angka[i]+list_angka[x]))
                list_finalnama.append(list_nama[i])
                # list_angka[i]=0
                # list_angka[x] = 0

    for i in range(0, len(list_nama)):
        if list_angka[i] != 0:
            list_final.append(math.sqrt(list_angka[i]))
            list_finalnama.append(list_nama[i])

    for i in range(0, len(list_final)):
        dict_dj2[list_finalnama[i]] = list_final[i]

    for key in dict_w:
        list_wnama.append(key)
        list_wangka.append(dict_w[key])

    for i in range(0, len(list_wnama)):
        for x in range(i+1, len(list_wnama)):
            if(list_wnama[i] == list_wnama[x][0:len(list_wnama[x])-8]):
                list_simnama.append(list_wnama[x][0:len(list_wnama[x])-8])
                list_simdoc.append(list_wnama[x][len(list_wnama[x])-7:])
                list_sim.append(list_wangka[x])
                list_wangka[x] = 0

    for i in range(0, len(list_wangka)):
        if(list_wangka[i] != 0):
            list_simq.append(list_wangka[i])
            list_simqnama.append(list_wnama[i])

    for i in range(0, len(list_sim)):
        for y in range(0, len(list_simq)):
            if(list_simnama[i] == list_simqnama[y]):
                list_finalsim.append(list_sim[i]*list_simq[y])

    for i in range(0, len(list_finalsim)):
        for x in range(i+1, len(list_simdoc)):
            if list_simdoc[i] == list_simdoc[x]:
                list_finalsim2.append(list_finalsim[i]+list_finalsim[x])
                list_finalsim[i] = 0
                list_finalsim[x] = 0
                list_finalsim2doc.append(list_simdoc[i])

    for i in range(0, len(list_finalsim)):
        if(list_finalsim[i] != 0):
            list_finalsim2.append(list_finalsim[i])
            list_finalsim2doc.append(list_simdoc[i])

    for i in range(0, len(list_finalsim2)):
        try:
            dict_sim[list_finalsim2doc[i]] = list_finalsim2[i] / \
                (dict_dj2[list_finalsim2doc[i]]*dict_dj2['q'])
        except:
            print("0")

    result_dictionary = dict()
    if limit > len(dict_sim):
        limit = len(dict_sim)

    counter = 0
    for i, j in dict_sim.items():
        counter += 1
        result_dictionary[i] = j
        if counter == limit:
            break

    stop = timeit.default_timer()

    return result_dictionary, stop-starts


def tfidf_ascending(kueri, limit):
    df_tfidf = pd.read_csv('Konstruksi Indeks.csv')
    doc = df_tfidf['Documents']
    term = df_tfidf['Term']
    df_tfidf
    max_range = len(term)

    query = kueri
    text, data = preprocess(query)

    z = 1
    count = 1
    N = 500
    plus = 0
    list_angka = list()
    list_nama = list()
    list_final = list()
    list_finalnama = list()

    list_wnama = list()
    list_wangka = list()
    list_simnama = list()
    list_simdoc = list()
    list_simq = list()
    list_simqnama = list()
    list_sim = list()
    list_finalsim = list()
    list_finalsim2 = list()
    list_finalsim2doc = list()

    dict_w = dict()
    dict_f = dict()
    dict_tf = dict()
    dict_idf = dict()
    dict_dj = dict()
    dict_dj2 = dict()
    dict_sim = dict()

    list_rank = list()
    counter = 0

    starts = timeit.default_timer()

    for q in data:
        if data[counter] == data[counter-1]:
            plus += 1
            dict_f[q] = plus
        else:
            plus = 1
            dict_f[q] = plus
        counter += 1

    for q in data:
        for i in range(0, max_range):
            ni = dict_f[q]
            start = 2
            end = 5
            count = 1
            if (term[i] == q):
                for y in range(0, int(len(doc[i])/7)):
                    if(int(len(doc[i])/7) != 1):
                        if(doc[i][start:end] != doc[i][(start-7):(end-7)]):
                            ni += 1
                            dict_f[q + ' doc ' + doc[i][start:end]] = count
                        else:
                            dict_f[q + ' doc' + doc[i][start:end]] = count
                            count += 1
                            ni = 1
                    else:
                        dict_f[q + ' doc ' + doc[i][start:end]] = count
                        ni += 1

                    start = start+7
                    end = end+7

                dict_idf[q] = math.log((1+N/ni), 10)

    for key in dict_f:
        dict_tf[key] = 1+math.log(dict_f[key], 10)

    for key in dict_tf:
        for key2 in dict_idf:
            if key[0:len(key)-8] == key2:
                dict_w[key] = dict_tf[key]*dict_idf[key2]
            elif key == key2:
                dict_w[key] = dict_tf[key]*dict_idf[key2]

    for key in dict_w:
        if(key[len(key)-7:len(key)-4] == 'doc'):
            dict_dj[key[len(key)-7:len(key)]] = (pow(dict_w[key], 2))
            list_angka.append(dict_dj[key[len(key)-7:len(key)]])
            list_nama.append(key[len(key)-7:len(key)])
        else:
            dict_dj[key] = pow(dict_w[key], 2)
            list_nama.append('q')
            list_angka.append(dict_dj[key])

    for i in range(0, len(list_nama)):
        for x in range(i+1, len(list_nama)):
            if (list_nama[i] == list_nama[x]):

                list_final.append(math.sqrt(list_angka[i]+list_angka[x]))
                list_finalnama.append(list_nama[i])
                # list_angka[i]=0
                # list_angka[x] = 0

    for i in range(0, len(list_nama)):
        if list_angka[i] != 0:
            list_final.append(math.sqrt(list_angka[i]))
            list_finalnama.append(list_nama[i])

    for i in range(0, len(list_final)):
        dict_dj2[list_finalnama[i]] = list_final[i]

    for key in dict_w:
        list_wnama.append(key)
        list_wangka.append(dict_w[key])

    for i in range(0, len(list_wnama)):
        for x in range(i+1, len(list_wnama)):
            if(list_wnama[i] == list_wnama[x][0:len(list_wnama[x])-8]):
                list_simnama.append(list_wnama[x][0:len(list_wnama[x])-8])
                list_simdoc.append(list_wnama[x][len(list_wnama[x])-7:])
                list_sim.append(list_wangka[x])
                list_wangka[x] = 0

    for i in range(0, len(list_wangka)):
        if(list_wangka[i] != 0):
            list_simq.append(list_wangka[i])
            list_simqnama.append(list_wnama[i])

    for i in range(0, len(list_sim)):
        for y in range(0, len(list_simq)):
            if(list_simnama[i] == list_simqnama[y]):
                list_finalsim.append(list_sim[i]*list_simq[y])

    for i in range(0, len(list_finalsim)):
        for x in range(i+1, len(list_simdoc)):
            if list_simdoc[i] == list_simdoc[x]:
                list_finalsim2.append(list_finalsim[i]+list_finalsim[x])
                list_finalsim[i] = 0
                list_finalsim[x] = 0
                list_finalsim2doc.append(list_simdoc[i])

    for i in range(0, len(list_finalsim)):
        if(list_finalsim[i] != 0):
            list_finalsim2.append(list_finalsim[i])
            list_finalsim2doc.append(list_simdoc[i])

    for i in range(0, len(list_finalsim2)):
        try:
            dict_sim[list_finalsim2doc[i]] = list_finalsim2[i] / \
                (dict_dj2[list_finalsim2doc[i]]*dict_dj2['q'])
        except:
            print("0")

    result_dictionary = dict()
    if limit > len(dict_sim):
        limit = len(dict_sim)

    counter = 0
    for i, j in dict_sim.items():
        counter += 1
        result_dictionary[i] = j
        if counter == limit:
            break

    stop = timeit.default_timer()

    return result_dictionary, stop-starts


def toprank(request):
    if request.method == 'POST':
        form = SearchForm2(request.POST)
        if form.is_valid() and len(form.cleaned_data.get('query')) != 0:
            query = form.cleaned_data.get('query')
            limit = form.cleaned_data.get('limit')
            user_input = query.lower()
            docs_results = tfidf(user_input, limit)

            times = docs_results[1]

            documents = docs_results[0]
            founded = len(documents)

            result_dict = list()
            results = []

            for key, value in documents.items():
                result_dict.append((key[4:], value))
                result_sorted = sorted(
                    result_dict, key=lambda x: x[1], reverse=True)

            for key, value in result_sorted:
                dictionary = dict()
                filename = 'Doc0' + key + '.xml'
                doc = xml.dom.minidom.parse('XML/' + filename)
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

            return render(request, 'gui/TopRank.html', {'results': results, 'times': times, 'founded': founded, 'limit': len(results), 'query': query})

    else:
        form = SearchForm2()
    return render(request, 'gui/TopRank.html')


def toprank_ascending(request):
    if request.method == 'POST':
        form = SearchForm2(request.POST)
        if form.is_valid() and len(form.cleaned_data.get('query')) != 0:
            query = form.cleaned_data.get('query')
            limit = form.cleaned_data.get('limit')
            user_input = query.lower()
            docs_results = tfidf_ascending(user_input, limit)

            times = docs_results[1]

            documents = docs_results[0]
            founded = len(documents)

            result_dict = list()
            results = []

            for key, value in documents.items():
                result_dict.append((key[4:], value))
                result_sorted = sorted(
                    result_dict, key=lambda x: x[0])

            for key, value in result_sorted:
                dictionary = dict()
                filename = 'Doc0' + key + '.xml'
                doc = xml.dom.minidom.parse('XML/' + filename)
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

            return render(request, 'gui/TopRankAscending.html', {'results': results, 'times': times, 'founded': founded, 'limit': len(results), 'query': query})

    else:
        form = SearchForm2()
    return render(request, 'gui/TopRankAscending.html')


def button_build_index(request):
    if request.is_ajax():
        message = "Noice"
        data = json.dumps(message)

        return HttpResponse(data, content_type='application/json')
    else:
        raise Http404


def button_build_index_tf_idf(request):
    if request.is_ajax():
        message = str(build_index_tf_idf())
        data = json.dumps('Time taken: ' + message)

        return HttpResponse(data, content_type='application/json')
    else:
        raise Http404


def button_build_index_language_model(request):
    if request.is_ajax():
        message = str(build_index_language_model())
        data = json.dumps('Time taken: ' + message)

        return HttpResponse(data, content_type='application/json')
    else:
        raise Http404
