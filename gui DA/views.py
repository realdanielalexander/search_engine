from dateutil import parser
import timeit
from .forms import SearchForm
from django.contrib.auth.forms import UserCreationForm
import xml.dom.minidom
import os
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib import messages

# Built search functions
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import pandas as pd
import numpy as np


def join_tags(tokens):
    for i in enumerate(tokens):

        index = i[0]
        y = i[1]
        width = len(y)

        if y[(width-1):width] is '.':
            tokens[index] = y[0:(width-1)]

        # if tokens[index][0:1] is '-' or tokens[index][1:2] is '-':
        #     tokens[index] = ''

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

    tokens_clean = [x for x in tokens if x !=
                    '' and x not in ".,!?'" and x not in '<>']

    return tokens_clean


def search(user_input):
    df_read = pd.read_csv("Konstruksi Indeks.csv")
    df_read_copy = df_read
    df_read_copy

    # first
    user_input = user_input.lower()
    user_input = word_tokenize(user_input)
    user_input = join_tags(user_input)

    # second
    indent = list()
    for i in user_input:
        indent.append(i)

    # third
    count = 0
    checker = True
    for i in indent:
        save = ''
        n = 0
        y = 0
        z = 1

        list_and = list()
        for x in df_read_copy['Term']:

            if i == 'and':
                if (indent[count-1] == df_read_copy['Term'].values[y]):
                    for v in df_read_copy['Term']:
                        if(indent[count+1] == df_read_copy['Term'].values[n]):
                            # print(indent[count-1] + ' ' + i + ' ' + v)
                            # print(df_read_copy['Documents'].values[y])
                            # print(df_read_copy['Documents'].values[n])
                            b = len(df_read_copy['Documents'].values[y])/7
                            c = len(df_read_copy['Documents'].values[n])/7
                            k = df_read_copy['Documents'].values[n][2:5]
                            l = df_read_copy['Documents'].values[n][9:12]
                            o = 2
                            j = 5
                            for m in range(max(int(b), int(c))):
                                if(df_read_copy['Documents'].values[y][o:j] == df_read_copy['Documents'].values[n][o:j]):
                                    list_and = df_read_copy['Documents'].values[y][o:j]
                                    o += 7
                                    j += 7
                                    print("['"+list_and+"']")
                        n += 1
                checker = False

            elif i == 'or':
                if (indent[count-1] == df_read_copy['Term'].values[y]):
                    for v in df_read_copy['Term']:
                        if(indent[count+1] == df_read_copy['Term'].values[n]):
                            # print(indent[count-1] + ' '+ i + ' ' + v + '\n')
                            # print(indent[count-1])
                            # print(df_read_copy['Documents'].values[y])
                            # print(v)
                            # print(df_read_copy['Documents'].values[n])
                            pass
                        n += 1
                checker = False
            y += 1
        count += 1
        if len(indent) != 1:
            if indent[z] == indent[z-1]:
                break

    if checker == True:
        for i in indent:
            y = 0
            for x in df_read_copy['Term']:
                if (i == df_read_copy['Term'].values[y]):
                    # print(i)
                    # print(df_read_copy['Documents'].values[y])
                    pass
                y += 1

    # fourth
    # print('indent:', indent)
    results = dict()
    for i in indent:
        y = 0
        for x in df_read_copy['Term']:
            if (i == df_read_copy['Term'].values[y]):
                # print(i)
                # print(df_read_copy['Documents'].values[y])
                # print(df_read_copy['Documents'].values[y].strip("[]"))
                list_res = df_read_copy['Documents'].values[y].strip(
                    "[]").replace("'", "").split(', ')

                results[i] = list_res
            y += 1
    return results

# SOUNDEX


def changeToNumber(text):
    changenumber = []
    texts = text
    text_list = list(texts)
    text_length = len(text_list)

    for x in text[1:text_length]:
        for y in x:
            if 'a' in y or 'e' in y or 'i' in y or 'o' in y or 'u' in y or 'h' in y or 'w' in y or 'y' in y:
                y = '0'
            elif 'b' in y or 'f' in y or 'p' in y or 'v' in y:
                y = '1'
            elif 'c' in y or 'g' in y or 'j' in y or 'k' in y or 'q' in y or 's' in y or 'x' in y or 'z' in y:
                y = '2'
            elif 'd' in y or 't' in y:
                y = '3'
            elif 'l' in y:
                y = '4'
            elif 'm' in y or 'n' in y:
                y = '5'
            elif 'r' in y:
                y = '6'
            changenumber.append(y)
    return changenumber


def changeZero(text):
    changezero = []

    for i in text:
        if i is '0':
            changezero.append(None)
        else:
            changezero.append(i)
    return changezero


def deleteNone(text):
    nones = []

    for val in text:
        if val != None:
            nones.extend(val)
    return nones


def listToString(s):
    str1 = ""
    return(str1.join(s))


def delete_consequent(s):
    list_s = s
    for i in range(len(s)-1):
        if list_s[i] == list_s[i+1]:
            list_s[i] = None
    return list_s


def soundex(text):
    texts = text.lower()
    text_list = list(texts)
    text_length = len(text_list)

    change_to_number_list = []

    change_to_number_list = changeToNumber(text_list)

    index_number = []

    # ambil huruf pertama
    index_number.extend(text_list[0:1])

    # menambahkan char berikutnya yang sudah diubah menjadi angka
    index_number.extend(change_to_number_list)

    change_zero = []
    change_zero = changeZero(index_number)

    deleted_consequent = delete_consequent(change_zero)
    none_value = []

    none_value = deleteNone(change_zero)

    results = listToString(none_value)

    return results


def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x, y] = min(matrix[x-1, y] + 1,
                                   matrix[x-1, y-1], matrix[x, y-1] + 1)
            else:
                matrix[x, y] = min(matrix[x-1, y] + 1,
                                   matrix[x-1, y-1] + 1, matrix[x, y-1] + 1)
    # print(matrix)
    return (matrix[size_x - 1, size_y - 1])

# Create your views here.


def search_and(dict_results):
    results = set()
    results.update(dict_results[list(dict_results.keys())[0]])

    for i, j in dict_results.items():
        results = set(j).intersection(results)
    results = sorted(list(results))
    return results


def search_or(dict_results):
    results = set()
    for i, j in dict_results.items():
        results.update(j)
    results = sorted(list(results))

    return results


def search_not(dict_results):
    total = ["{0:0=3d}".format(i) for i in range(1, 501)]
    results = set()
    for i, j in dict_results.items():
        results.update(j)
    results = sorted(list(results))
    results_not = [i for i in total if i not in results]
    print('res', results_not)
    return results_not

# Preprocessing


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def preprocess(text_process):
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    text_process = text_process.lower()
    tokens = word_tokenize(text_process)
    tokens_clean = join_tags(tokens)
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in tokens_clean if not w in stop_words]
    stemmed_words = [ps.stem(w) for w in filtered_words]
    lemmatized_words = [lemmatizer.lemmatize(
        w, get_wordnet_pos(w)) for w in stemmed_words]
    stemmed_text = ' '.join(stemmed_words)
    lemmatized_text = ' '.join(lemmatized_words)
    #clean_text = lemmatized_text
    clean_text = lemmatized_text
    return clean_text


def home(request):
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid() and len(form.cleaned_data.get('query')) != 0:
            start = timeit.default_timer()
            results = []
            query = form.cleaned_data.get('query')
            if query[:9].lower() == 'accessed:':
                user_input = query.lower()
                user_input = word_tokenize(user_input)
                user_input = join_tags(user_input)
                date = parser.parse(user_input[1])
                query = date.strftime(" %d %b %Y").replace(' 0', ' ')
                print('qq: ', query)
                query = query.strip()
                list_q = query.split()
                query = 'Accessed: ' + '-'.join(list_q)
                print('date: ', query)

            docs = search(query)

            # if len(docs) == 0:
            #     return render(request, 'gui/index.html')
            print('query:', docs)
            # initalize numbers variable
            numbers = set()

            # get correction
            correction = form.cleaned_data.get('correction')
            print('correction: ', correction)

            user_input = query.lower()
            user_input = word_tokenize(user_input)
            user_input = join_tags(user_input)
            print('user input: ', user_input)
            if 'soundex' in correction:
                user_input = query.lower()
                user_input = word_tokenize(user_input)
                user_input = join_tags(user_input)
                print('user input :', user_input)
                print('preprocessed: ', preprocess(query))
                soundex_codes = [soundex(i) for i in user_input]
                print(soundex_codes)

                df_read = pd.read_csv("Konstruksi Indeks.csv")
                for index, row in df_read.iterrows():
                    if soundex(row['Term']) in soundex_codes:
                        # numbers.update(row['Documents'])
                        numbers.update(row['Documents'].strip(
                            "[]").replace("'", "").split(', '))

            if 'levenshtein' in correction:
                user_input = query.lower()
                user_input = word_tokenize(user_input)
                user_input = join_tags(user_input)
                print('user input :', user_input)
                print('preprocessed: ', preprocess(query))
                threshold = form.cleaned_data.get('threshold')
                df_read = pd.read_csv("Konstruksi Indeks.csv")
                for index, row in df_read.iterrows():
                    for i in user_input:
                        if levenshtein(row['Term'], i) <= threshold:
                            numbers.update(row['Documents'].strip(
                                "[]").replace("'", "").split(', '))

            print('nums:', numbers)
            # get method (and, or, not)
            method = form.cleaned_data.get('method')
            print('method:', method)
            if method == 'and':
                numbers.update(search_and(docs))
            elif method == 'or':
                numbers.update(search_or(docs))
            elif method == 'not':
                numbers.update(search_not(docs))

            numbers = sorted(list(numbers))
            print(numbers)

            display = form.cleaned_data.get('display')
            print('display: ', display)
            # read xml
            for i in numbers:
                dictionary = dict()
                filename = 'Doc0' + i + '.xml'
                print('filename: ', filename)
                if display == 'original':
                    doc = xml.dom.minidom.parse('XML/' + filename)
                    title = doc.getElementsByTagName('TITLE')
                    title = title[0].firstChild.nodeValue

                    date = doc.getElementsByTagName('DATE')
                    date = date[0].firstChild.nodeValue
                    body = doc.getElementsByTagName('BODY')
                    body = body[0].firstChild.nodeValue
                elif display == 'clean':
                    doc = xml.dom.minidom.parse('Clean/' + filename)
                    title = doc.getElementsByTagName('title')
                    title = title[0].firstChild.nodeValue
                    date = doc.getElementsByTagName('date')
                    date = date[0].firstChild.nodeValue
                    body = doc.getElementsByTagName('body')
                    body = body[0].firstChild.nodeValue
                # print(date)
                dictionary['title'] = title
                dictionary['date'] = date
                dictionary['body'] = body
                results.append(dictionary)
            # for term, documents in docs.items():
            #     for no_doc in documents:
            #         dictionary = dict()
            #         filename = 'Doc0' + no_doc + '.xml'
            #         doc = xml.dom.minidom.parse('Clean/' + filename)
            #         title = doc.getElementsByTagName('title')
            #         title = title[0].firstChild.nodeValue
            #         body = doc.getElementsByTagName('body')
            #         body = body[0].firstChild.nodeValue
            #         dictionary['title'] = title
            #         dictionary['body'] = body
            #         results.append(dictionary)
            stop = timeit.default_timer()
            return render(request, 'gui/index.html', {'results': results, 'count': len(results), 'time': stop-start, 'method': method, 'correction': correction, 'display': display, 'query': query})
    else:
        form = SearchForm()
    return render(request, 'gui/index.html')
