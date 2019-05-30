from __future__ import division
import math
import nltk
import string
import os
import re
import collections
import random
from collections import Counter
from nltk.stem.porter import PorterStemmer
import collections
from decimal import *
from itertools import islice
import gensim, logging
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from scipy import spatial
# random
from random import shuffle

fullAnnot = "fullAnnotation.conf"
objURLS = "objectURLS.conf"
N = 10

#tokenize = lambda doc: doc.split(" ")

def objectURLs() :
  objLinks = {}
  with open(objURLS,'r') as f:
   for line in f:
      l = line.split(",")
      l2 = line.replace(l[0]+ ",",'')
      l3 = l2.replace('\n','')
      objLinks[l[0]] = l3
  return objLinks


def objectNames() :
  fullAnnotations = {}
  with open(fullAnnot,'r') as f:
   for line in f:
      l = line.split(",")
      l2 = line.replace(l[0]+ ",",'')
      l3 = l2.replace('\n','')
      fullAnnotations[l[0]] = l3
  return fullAnnotations

def getDocuments(fName):
   #fname is the name of the raw preprocessed file
   instSentences = {}
   with open(fName, 'r') as f:
    for line in f:
     l = line.split(",")
     l2 = line.replace(l[0]+ ",",'')
     l3 = l2.replace('\n','')
     if(line != "" and l3 != "") :
      if l[0] in instSentences.keys():
         sent = instSentences[l[0]]
         sent += " " + l3
         instSentences[l[0]] = sent
      else:
         instSentences[l[0]] = l3
   sortedinstSentences = collections.OrderedDict(sorted(instSentences.items()))
#   keys = list(instSentences.keys())
#   random.shuffle(keys)
#   random.shuffle(keys)

#   sortedinstSentences = {}
#   for key in keys:
#       sortedinstSentences[key] = instSentences[key]
#   print sortedinstSentences.keys()
   return sortedinstSentences

def getDocsForTest(arTokens,fName):
  """
  This function reads in the preprocessed file, and grabs the
  lines that are related to the tokens that are being tested.
  """
  instSentences = {}
  with open(fName, 'r') as f:
   for line in f:
    l = line.split(",")
    if l[0] in arTokens:
     l2 = line.replace(l[0]+ ",",'')
<<<<<<< HEAD
     l3 = l2.replace('\n','').replace("\r","")
=======
     l3 = l2.replace('\n','')
#     l3 = re.sub('[^A-Za-z0-9\ ]+', '', l3)
#     l3 = l3.lower()
>>>>>>> 02ac5e1223607178690fe84aed74d7c57522ed55
     if(line != "" and l3 != "") :
      if l[0] in instSentences.keys():
         sent = instSentences[l[0]]
         sent += " " + l3
         instSentences[l[0]] = sent
      else:
         instSentences[l[0]] = l3
   sortedinstSentences = collections.OrderedDict(sorted(instSentences.items()))
   return sortedinstSentences

def sentenceToWordLists(docs):
   docLists = []
   for key in docs.keys():
      sent = docs[key]
      wLists = sent.split(" ")
      docLists.append(wLists)
   return docLists

def sentenceToWordDicts(docs):
<<<<<<< HEAD
   """
   This function just turns a dictionary of strings into a dictionary
   where the strings are now a list of words.
   """
=======
>>>>>>> 02ac5e1223607178690fe84aed74d7c57522ed55
   docDicts = {}
   for key in docs.keys():
      sent = docs[key]
      wLists = sent.split(" ")
      docDicts[key] = wLists
   return docDicts

def findtfIDFLists(docLists):
   arWords = []
   for dList in docLists:
      arWords.extend(set(dList))
   arIDF = {}
   arIDFCount = Counter(arWords)
   for x in arIDFCount.keys():
      arIDF[x] = math.log(len(docLists)/arIDFCount[x])

   tdIDFLists = []
   for dList in docLists:
      dictC = Counter(dList)
      tfidfValues = []
      for word in dList:
         tfidfValues.append(dictC[word] * arIDF[word])
      tdIDFLists.append(tfidfValues)
   return tdIDFLists

def findTopNtfidfterms(docLists,tfidfLists,N):
   topTFIDFWordLists = []
   for i in range(len(docLists)):
      dList = docLists[i]
      tList = tfidfLists[i]
      dTFIDFMap = {}
      for j in range(len(dList)):
          dTFIDFMap[dList[j]] = tList[j]
<<<<<<< HEAD

=======

>>>>>>> 02ac5e1223607178690fe84aed74d7c57522ed55
      stC = sorted(dTFIDFMap.items(), key=lambda x: x[1])
      lastpairs = stC[len(stC) - N  :]
      vals = []
      for jj in lastpairs:
         vals.append(jj[0])
      topTFIDFWordLists.append(vals)
   return topTFIDFWordLists

class LabeledLineSentence(object):
    def __init__(self,docLists,docLabels):
        self.docLists = docLists
        self.docLabels = docLabels

    def __iter__(self):
        for index, arDoc in enumerate(self.docLists):
            yield LabeledSentence(arDoc, [self.docLabels[index]])

    def to_array(self):
        self.sentences = []
        for index, arDoc in enumerate(self.docLists):
            self.sentences.append(LabeledSentence(arDoc, [self.docLabels[index]]))
        return self.sentences
<<<<<<< HEAD

=======

>>>>>>> 02ac5e1223607178690fe84aed74d7c57522ed55
    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences

def square_rooted(x):
    return round(math.sqrt(sum([a*a for a in x])),3)

def cosine_similarity(x,y):
   numerator = sum(a*b for a,b in zip(x,y))
   denominator = square_rooted(x)*square_rooted(y)
   return round(numerator/float(denominator),3)

def doc2Vec(docs):
<<<<<<< HEAD
  """
  Ths function takes in the instance:all descriptions dictionary.

  """
  docLabels = []
  docNames = docs.keys()
  for key in docs.keys():
    ar = key.split("/")
    docLabels.append(key)
=======
  docLabels = []
  docNames = docs.keys()
  for key in docs.keys():
   ar = key.split("/")
   docLabels.append(key)
>>>>>>> 02ac5e1223607178690fe84aed74d7c57522ed55
  docLists = sentenceToWordLists(docs)
  docDicts = sentenceToWordDicts(docs)
  sentences = LabeledLineSentence(docLists,docLabels)
  model = Doc2Vec(min_count=1, window=10, size=2000, sample=1e-4, negative=5, workers=8)
  model.build_vocab(sentences.to_array())
  token_count = sum([len(sentence) for sentence in sentences])
  for epoch in range(10):
    model.train(sentences.sentences_perm(),total_examples = token_count,epochs=model.iter)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no deca
    model.train(sentences.sentences_perm(),total_examples = token_count,epochs=model.iter)
  degreeMap = {}
  angles = []
<<<<<<< HEAD
  #loop over all pairs of instance
=======
>>>>>>> 02ac5e1223607178690fe84aed74d7c57522ed55
  for i , item1 in enumerate(docLabels):
   fDoc = model.docvecs[docLabels[i]]

   cInstMap = {}
   cInstance = docLabels[i]
   for j,item2 in enumerate(docLabels):
     tDoc = model.docvecs[docLabels[j]]
<<<<<<< HEAD
     #get the similarity between instances
     cosineVal = max(-1.0,min(1.0,cosine_similarity(fDoc,tDoc)))
=======
     cosineVal = cosine_similarity(fDoc,tDoc)
>>>>>>> 02ac5e1223607178690fe84aed74d7c57522ed55
     tInstance = docLabels[j]
     cValue = math.degrees(math.acos(cosineVal))
     cInstMap[tInstance] = cValue
     angles.append(cValue)
   degreeMap[cInstance] = cInstMap
  maxAngle = max(angles)
  thresh5th = maxAngle
  thresh4th = maxAngle / 3
<<<<<<< HEAD
  #  thresh4th = 50.0
  negMaps = {}
  #this appears to return the similarity metric between an intance and every other instance
  #in order with the most similar instances first

  for k,v in degreeMap.items() :
   #print "#######################"
   #print k,":",v
   negMaps[k] = []
   ss = sorted(v.items(), key=lambda x: x[1])

   #This choose the second two thirds of the list as the most different
   sPoint = int(len(ss)/3)
   ssNew = ss[sPoint:]
   negMaps[k] = ssNew
   #for itemSS in ssNew:
   #    negMaps[k].append(itemSS[0])

=======
#  thresh4th = 50.0
  negMaps = {}
  for k,v in degreeMap.items() :
   negMaps[k] = []
   ss = sorted(v.items(), key=lambda x: x[1])
   sPoint = int(len(ss)/3)
   ssNew = ss[sPoint:]
   for itemSS in ssNew:
       negMaps[k].append(itemSS[0])
>>>>>>> 02ac5e1223607178690fe84aed74d7c57522ed55
  return negMaps
