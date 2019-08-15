import random

from gensim.models.doc2vec import LabeledSentence

class LabeledLineSentence:
    def __init__(self, docLists, docLabels):
        self.docLists = docLists
        self.docLabels = docLabels
        self.sentences = []

    def __iter__(self):
        for index, arDoc in enumerate(self.docLists):
            yield LabeledSentence(arDoc, [self.docLabels[index]])

    def to_array(self):
        self.sentences = []
        for index, arDoc in enumerate(self.docLists):
            self.sentences.append(LabeledSentence(arDoc, [self.docLabels[index]]))
        return self.sentences

    # PAT TODO: if this is called before to_array it will return []
    def sentences_perm(self):
        random.shuffle(self.sentences)
        return self.sentences
