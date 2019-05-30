# -*- coding: utf-8 -*-
"""
This file preprocesses the description files according to the specified arguments
The user can add "stop", "lemm", or "stemm" as arguments to add stop word removal,
lemmatization, or stemming respecively.
The first two arguments should be <original file> <language>
"""

import sys
import math
import nltk
import string
import os
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import codecs
reload(sys)  # This is a bit of a hack to avoid encoding errors
sys.setdefaultencoding('UTF8')

lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()

''' Lightweight Hindi stemmer
Copyright © 2010 Luís Gomes <luismsgomes@gmail.com>.

Implementation of algorithm described in

    A Lightweight Stemmer for Hindi
    Ananthakrishnan Ramanathan and Durgesh D Rao
    http://computing.open.ac.uk/Sites/EACLSouthAsia/Papers/p6-Ramanathan.pdf

    @conference{ramanathan2003lightweight,
      title={{A lightweight stemmer for Hindi}},
      author={Ramanathan, A. and Rao, D.},
      booktitle={Workshop on Computational Linguistics for South-Asian Languages, EACL},
      year={2003}
    }

Ported from HindiStemmer.java, part of of Lucene.
'''

suffixes = {
    1: ["ो", "े", "ू", "ु", "ी", "ि", "ा"],
    2: ["कर", "ाओ", "िए", "ाई", "ाए", "ने", "नी", "ना", "ते", "ीं", "ती", "ता", "ाँ", "ां", "ों", "ें"],
    3: ["ाकर", "ाइए", "ाईं", "ाया", "ेगी", "ेगा", "ोगी", "ोगे", "ाने", "ाना", "ाते", "ाती", "ाता", "तीं", "ाओं", "ाएं", "ुओं", "ुएं", "ुआं"],
    4: ["ाएगी", "ाएगा", "ाओगी", "ाओगे", "एंगी", "ेंगी", "एंगे", "ेंगे", "ूंगी", "ूंगा", "ातीं", "नाओं", "नाएं", "ताओं", "ताएं", "ियाँ", "ियों", "ियां"],
    5: ["ाएंगी", "ाएंगे", "ाऊंगी", "ाऊंगा", "ाइयाँ", "ाइयों", "ाइयां"],
}

def hi_stem(word):
    for L in 5, 4, 3, 2, 1:
        if len(word) > L + 1:
            for suf in suffixes[L]:
                if word.endswith(suf):
                    return word[:-L]
    return word


def filteringSentences(fName, lemm=True, stemm=True, stop=True,lan = "english"):
  if stemm:
    if lan in ["spanish","english"]:
        stemmer = SnowballStemmer(lan)
  with codecs.open(fName, 'r',encoding="utf-8") as f:
    f_start = fName.replace(".conf","")
    ending = ""
    if stemm:
        ending += "_stemmed"
    if lemm:
        ending += "_lemmed"
    if stop:
        ending += "_stop"
        if lan in ["spanish","english"]:
            stop_words = set(stopwords.words(lan))


        else:
            stop_words = []
    if not(stemm) and not(lemm):
        ending += "_raw"
    ending += ".conf"
    with codecs.open(f_start+ending,"w",encoding="utf-8") as write_file:
      for line in f:
        l = line.split(",")
        l2 = line.replace(l[0]+ ",",'')
        l3 = l2.replace("-"," ")
        #l3 = re.sub('[^A-Za-z0-9\ ]+', '', l3)
        for rem in ["\n","\t","\r",".","?","!",u'¿',u'¡',u'।',u"\u0964","|"]:
           l3 = l3.replace(rem,"")
        for pun in string.punctuation:
           l3 = l3.replace(pun,"")
        l3 = l3.lower()
        while "  " in l3:
            l3 = l3.replace("  "," ")

        if(line != "" and l3 != "") :
            filtered_sentence = l3.split(" ")
        if stop:
            filtered_sentence = [w for w in filtered_sentence if not w in stop_words]
        if stemm:
            if lan == "hindi":
                filtered_sentence = [hi_stem(w) for w in filtered_sentence]
            else:
                filtered_sentence = [stemmer.stem(w) for w in filtered_sentence]

        if lemm:
            filtered_sentence = [lemmatizer.lemmatize(w) for w in filtered_sentence]
        if len(filtered_sentence) > 0:
            out_string = l[0]+ "," + " ".join(filtered_sentence)
            write_file.write(out_string+"\n")
            #print out_string
    print lan,lemm,stemm,stop


if __name__ == "__main__":
    st = sys.argv
    fName = str(st[1])

    language = str(st[2])

    if "lemm" in st:
        lemm=True
    else:
        lemm=False

    if "stemm" in st:
        stemm=True
    else:
        stemm=False

    if "stop" in st:
        stop=True
    else:
        stop=False
    #fName = "Batch_3166944_batch_results.csv"
    #fName = "Batch_3169679_batch_results.csv"
    #fName = "Batch_3178122_batch_results.csv"
    filteringSentences(fName,lemm,stemm,stop,language)
