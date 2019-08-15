import collections
import math
import pickle

import numpy as np
from gensim.models import Doc2Vec

from labeled_line_sentence import LabeledLineSentence

resultDir = ARGS.resDir

class NegSampleSelection:
    """ Class to bundle negative example generation functions and variables. """

    def __init__(self, docs):
        """
        Initialization function for NegSampleSelection class
        Args: Documents dictionary where key is object instance and value
            is object annotation
        Returns: Nothing
        """
        self.docs = collections.OrderedDict(sorted(docs.items()))

    def sentenceToWordLists(self):
        docLists = []
        docs = self.docs
        for key in docs.keys():
            sent = docs[key]
            wLists = sent.split(" ")
            docLists.append(wLists)
        return docLists

    def sentenceToWordDicts(self):
        docs = self.docs
        docDicts = {}
        for key in docs.keys():
            sent = docs[key]
            wLists = sent.split(" ")
            docDicts[key] = wLists
        return docDicts

    def square_rooted(self, x):
        return round(math.sqrt(sum([a * a for a in x])), 3)

    def cosine_similarity(self, x, y):
        numerator = sum(a * b for a, b in zip(x, y))
        denominator = self.square_rooted(x) * self.square_rooted(y)
        return round(numerator / float(denominator), 3)

    # PAT TODO: check this for correct indentation against repo
    def generateNegatives(self):
        docs = self.docs
        docNames = docs.keys()
        docLists = self.sentenceToWordLists()
        docLabels = []
        for key in docNames:
            ar = key.split("/")
            docLabels.append(ar[1])
        sentences = LabeledLineSentence(docLists, docLabels)
        model = Doc2Vec(min_count=1, window=10, size=2000, sample=1e-4, negative=5, workers=8)

        model.build_vocab(sentences.to_array())
        token_count = sum([len(sentence) for sentence in sentences])
        for epoch in range(10):
            model.train(sentences.sentences_perm(), total_examples=token_count, epochs=model.iter)
            model.alpha -= 0.002 # decrease the learning rate
            model.min_alpha = model.alpha # fix the learning rate, no deca
            model.train(sentences.sentences_perm(), total_examples=token_count, epochs=model.iter)

        degreeMap = {}
        for i in range(len(docLabels)):
            fDoc = model.docvecs[docLabels[i]]
            cInstMap = {}
            cInstance = docNames[i]
            for j in range(len(docLabels)):

                tDoc = model.docvecs[docLabels[j]]
                cosineVal = max(-1.0, min(self.cosine_similarity(fDoc, tDoc), 1.0))

                try:
                    cValue = math.degrees(math.acos(cosineVal))
                except:
                    print("ERROR: invalid cosine value")
                    print(cosineVal)
                    print(fDoc)
                    print(tDoc)
                    exit()
                tInstance = docNames[j]
                cInstMap[tInstance] = cValue
            degreeMap[cInstance] = cInstMap
        negInstances = {}
        for k in np.sort(degreeMap.keys()):
            v = degreeMap[k]
            ss = sorted(v.items(), key=lambda x: x[1])
            sentAngles = ""
            for item in ss:
                if item[0] != k:
                    sentAngles += item[0]+"-"+str(item[1])+","
            sentAngles = sentAngles[:-1]
            negInstances[k] = sentAngles

        # pickle negative examples for later use
        with open('NegExamples_'+ resultDir + '.pickle', 'wb') as handle:
            pickle.dump(negInstances, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return negInstances
