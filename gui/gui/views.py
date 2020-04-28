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
#                 save = tokens[index][hit+1:]
#                 tokens[index] = tokens[index][0:hit] + ''
#                 tokens[index+1] = save
#                 print(save)
#                 print(tokens[index])
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
    for index, i in enumerate(os.listdir('C:/Users/Asus/Downloads/Git/search_engine/Data')):
        try:
            filename = i
            f = open('C:/Users/Asus/Downloads/Git/search_engine/Data/' + filename, 'r')
            text = f.read()
            f.close()

            clean_text, clean_words = preprocess(text)
            clean_words_list.extend(clean_words)

    #         print(clean_words)
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
    #         target = 'Clean/' + filename[:-3] +'xml'
    #         new_file = open(target, 'w+')
    #         new_file.write(clean_text)
    #         new_file.close()
        except Exception as e:
            print(str(e))

#     print(doc_freqs)
    df_doc_freqs = pd.DataFrame(doc_freqs.items(), columns = ['Document', 'Frequency'])
    df_doc_freqs.to_csv('Document Frequencies.csv')

    for i in sorted(dict_term_doc.keys()):
        dict_term_doc_final[i] = dict_term_doc[i]
    df_term_doc = pd.DataFrame(dict_term_doc_final.items(), columns = ['Term', 'Documents'])
    df_term_doc.to_csv('Frequencies.csv')
    
    stop = timeit.default_timer()
    return stop-start

def language_model(query, limit, lambd):
    clean_query_text, clean_query_words = preprocess(query)
    print(clean_query_words)
#     lambd = 1/4
#     limit = 10
    print(clean_query_text)
    result_dict = list()

    df_doc_freq = pd.read_csv('C:/Users/Asus/Downloads/Git/search_engine/Document Frequencies.csv')
    df = pd.read_csv('C:/Users/Asus/Downloads/Git/search_engine/Frequencies.csv')
    global_freq = int(open("C:/Users/Asus/Downloads/Git/search_engine/global_frequency.txt", "r").read())
#     print(global_freq)

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
            numbers = []
            score = []
            
            # test = docs_results

            for i in documents:
                numbers.append(i[0])
                score.append(i[1])

            results = []

            # read xml
            for key,value in documents:
                dictionary = dict()
                filename = 'Doc0' + key + '.xml'
                print('filename: ', filename)
                # if display == 'original':
                doc = xml.dom.minidom.parse('C:/Users/Asus/Downloads/Git/search_engine/XML/' + filename)
                title = doc.getElementsByTagName('TITLE')
                title = title[0].firstChild.nodeValue

                date = doc.getElementsByTagName('DATE')
                date = date[0].firstChild.nodeValue
                body = doc.getElementsByTagName('BODY')
                body = body[0].firstChild.nodeValue
                # elif display == 'clean':
                #     doc = xml.dom.minidom.parse('Clean/' + filename)
                #     title = doc.getElementsByTagName('title')
                #     title = title[0].firstChild.nodeValue
                #     date = doc.getElementsByTagName('date')
                #     date = date[0].firstChild.nodeValue
                #     body = doc.getElementsByTagName('body')
                #     body = body[0].firstChild.nodeValue
                # print(date)
                dictionary['document_no'] = key
                dictionary['score'] = value
                dictionary['title'] = title
                dictionary['date'] = date
                dictionary['body'] = body
                results.append(dictionary)

            return render(request, 'Index.html', {'results' : results, 'times': times, 'founded': founded,  'query': query, 'limit' : limit, 'lambdaa' : lambdaa, })
    else:
        # context['query'] = request.POST['queries']
        form = SearchForm()
    return render(request, 'Index.html')


# TF IDF

def tfidf(kueri):
    df_tfidf = pd.read_csv('C:/Users/Asus/Downloads/Git/search_engine/Konstruksi Indeks.csv')
    doc = df_tfidf['Documents']
    term = df_tfidf['Term']
    df_tfidf
    max_range = len(term)


#     query = 'chairman closed the last'
    query = kueri
    data = query.split()
    z=1
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

    start = timeit.default_timer()

    for q in data:
        if data[counter]==data[counter-1]:
            plus+=1
            dict_f[q] = plus
        else:
            plus = 1
            dict_f[q] = plus
        counter+=1

    for q in data:
        for i in range (0,max_range):
            ni = dict_f[q]
            start = 2
            end = 5
            count = 1
            if (term[i] == q):
                for y in range (0,int(len(doc[i])/7)):
                    if(int(len(doc[i])/7)!=1):
                        if(doc[i][start:end] != doc[i][(start-7):(end-7)]):
                            ni+=1
                            dict_f[q +' doc ' + doc[i][start:end]]=count
                        else:
                            dict_f[q + ' doc' + doc[i][start:end]]=count
                            count+=1
                            ni=1
                    else:
                        dict_f[q +' doc ' + doc[i][start:end]]=count
                        ni+=1

                    start = start+7
                    end = end+7

                dict_idf[q] = math.log((1+N/ni),10)

    for key in dict_f:
        dict_tf[key] = 1+math.log(dict_f[key],10)


    for key in dict_tf:
        for key2 in dict_idf:
            if key[0:len(key)-8] == key2:
                dict_w[key] = dict_tf[key]*dict_idf[key2]
            elif key == key2:
                dict_w[key] = dict_tf[key]*dict_idf[key2]

    for key in dict_w:
        if(key[len(key)-7:len(key)-4] == 'doc' ):
            dict_dj[key[len(key)-7:len(key)]] = (pow(dict_w[key],2))
            list_angka.append(dict_dj[key[len(key)-7:len(key)]])
            list_nama.append(key[len(key)-7:len(key)])
        else:
            dict_dj[key] = pow(dict_w[key],2)
            list_nama.append('q')
            list_angka.append(dict_dj[key])

    for i in range(0,len(list_nama)):
        for x in range(i+1,len(list_nama)):
            if (list_nama[i]==list_nama[x]):

                list_final.append(math.sqrt(list_angka[i]+list_angka[x]))
                list_finalnama.append(list_nama[i])
                list_angka[i]=0
                list_angka[x] = 0

    for i in range(0,len(list_nama)):
        if list_angka[i]!=0:
            list_final.append(math.sqrt(list_angka[i]))
            list_finalnama.append(list_nama[i])

    for i in range(0,len(list_final)):
        dict_dj2[list_finalnama[i]] = list_final[i]


    for key in dict_w:
        list_wnama.append(key)
        list_wangka.append(dict_w[key])

    for i in range(0,len(list_wnama)):
        for x in range(i+1,len(list_wnama)):
            if(list_wnama[i]==list_wnama[x][0:len(list_wnama[x])-8]):
                list_simnama.append(list_wnama[x][0:len(list_wnama[x])-8])
                list_simdoc.append(list_wnama[x][len(list_wnama[x])-7:])
                list_sim.append(list_wangka[x])
                list_wangka[x] = 0

    for i in range(0,len(list_wangka)):
        if(list_wangka[i]!=0):
            list_simq.append(list_wangka[i])
            list_simqnama.append(list_wnama[i])

    for i in range(0,len(list_sim)):
        for y in range(0,len(list_simq)):
            if(list_simnama[i]==list_simqnama[y]):
                list_finalsim.append(list_sim[i]*list_simq[y])

    for i in range(0,len(list_finalsim)):
        for x in range(i+1,len(list_simdoc)):
            if list_simdoc[i]==list_simdoc[x]:
                list_finalsim2.append(list_finalsim[i]+list_finalsim[x])
                list_finalsim[i] = 0
                list_finalsim[x] = 0
                list_finalsim2doc.append(list_simdoc[i])

    for i in range(0,len(list_finalsim)):
        if(list_finalsim[i]!=0):
            list_finalsim2.append(list_finalsim[i])
            list_finalsim2doc.append(list_simdoc[i])

    for i in range(0,len(list_finalsim2)):
        dict_sim[list_finalsim2doc[i]] = list_finalsim2[i]/(dict_dj2[list_finalsim2doc[i]]*dict_dj2['q'])

    stop = timeit.default_timer()

#     print('Ranking :')
#     for key,value in dict_sim.items():
#         print(key + ": " ,value)
    return dict_sim, stop-start
    
def toprank(request):
    if request.method == 'POST':
        form = SearchForm2(request.POST)
        if form.is_valid() and len(form.cleaned_data.get('query')) != 0:
            query = form.cleaned_data.get('query')
            user_input = query.lower()

            docs_results = tfidf(user_input)
            founded = len(docs_results[0])
            times = docs_results[1]
            
            documents = docs_results[0]
            founded = len(documents)
                
            results = []
            for key,value in documents.items():
                dictionary = dict()
                filename = 'Doc0' + key[4:] + '.xml'
                print('filename: ', filename)
                # if display == 'original':
                doc = xml.dom.minidom.parse('C:/Users/Asus/Downloads/Git/search_engine/XML/' + filename)
                title = doc.getElementsByTagName('TITLE')
                title = title[0].firstChild.nodeValue

                date = doc.getElementsByTagName('DATE')
                date = date[0].firstChild.nodeValue
                body = doc.getElementsByTagName('BODY')
                body = body[0].firstChild.nodeValue
                # elif display == 'clean':
                #     doc = xml.dom.minidom.parse('Clean/' + filename)
                #     title = doc.getElementsByTagName('title')
                #     title = title[0].firstChild.nodeValue
                #     date = doc.getElementsByTagName('date')
                #     date = date[0].firstChild.nodeValue
                #     body = doc.getElementsByTagName('body')
                #     body = body[0].firstChild.nodeValue
                # print(date)
                dictionary['document_no'] = key[4:]
                dictionary['score'] = value
                dictionary['title'] = title
                dictionary['date'] = date
                dictionary['body'] = body
                results.append(dictionary)
            
            # return render(request, 'TopRank.html', {'results' : results, 'times': times, 'founded': founded, 'query': query,})
            return render(request, 'TopRank.html', {'results' : results, 'times': times, 'founded': founded, 'queries': results,})
    else:
        form = SearchForm2()
    return render(request, 'TopRank.html')