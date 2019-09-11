from collections import defaultdict

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np

def cosine_similarity(v1, v2):
    """
    Calculates cosine similarity between two vectors
    Code from: http://danushka.net/lect/dm/Numpy-basics.html
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class DataSet:
    def __init__(self, annotation_file, minimum_token_count=2, cutoff=0.005):
        self._data = None
        self._pos_instances_to_tokens = None
        self._neg_instances_to_tokens = None
        self._pos_tokens_to_instances = None
        self._neg_tokens_to_instances = None

        self._read_annotations(annotation_file)
        self._extract_and_set_positives(minimum_token_count)
        self._extract_and_set_negatives(cutoff)

    @property
    def data(self):
        return self._data

    @property
    def pos_instances_to_tokens(self):
        return self._pos_instances_to_tokens

    @property
    def pos_tokens_to_instances(self):
        return self._pos_tokens_to_instances

    @property
    def neg_instances_to_tokens(self):
        return self._neg_instances_to_tokens

    @property
    def neg_tokens_to_instances(self):
        return self._neg_tokens_to_instances

    @property
    def tokens(self):
        tokens = []
        for instance in self.data:
            tokens += self.data[instance]

        # return a list of unique tokens
        return list(set(tokens))

    def _read_annotations(self, annotation_file):
        """
        Parse the annotation file into a python dict
        {
            instance1: [tokens1],
            instance2: [tokens2],
            ...
        }

        "default" annotation_file
        conf_files/UW_english/UW_AMT_description_documents_per_image_nopreproc_stop_raw.conf

        :param str annotation_file: File path to amazon mechanical turk annotation file
        """
        data = {}
        with open(annotation_file, 'r') as annotations:
            for line in annotations:
                instance, tokens = line.strip().split(',')
                data[instance] = tokens.split(' ')

        self._data = data

    def _extract_and_set_positives(self, minimum_token_count):
        """
        Finds positive token examples of the instances in the annotation set.
        A positive example is an token that has described an instance more times
        than a minimum threshold

        Sets the relevant instance variables

        PAT TODO: In Nisha's paper this was done with tf-idf which this is no longer doing
            should it go back to tf-idf?

        :param int minimum_token_count: This controls the minimum number of times
            a token has to appear in descriptions for an instance before the instance
            is deemed to be a positive example of this token
        """
        instances_to_positive = defaultdict(list)

        for instance, tokens in self.data.items():
            for token in self.tokens:
                # Using a list with "not in" instead of a set so we can iterate later
                if token in tokens and tokens.count(token) >= minimum_token_count:
                    instances_to_positive[instance].append(token)

        self._pos_instances_to_tokens = dict(instances_to_positive)

        # Invert the dict to map tokens -> instances
        pos_tokens_to_instances = defaultdict(list)
        for instance, tokens in self._pos_instances_to_tokens.items():
            for token in tokens:
                pos_tokens_to_instances[token].append(instance)

        self._pos_tokens_to_instances = dict(pos_tokens_to_instances)

    def _extract_and_set_negatives(self, cutoff):
        """
        Finds negative instance examples of the tokens in the annotation set.
        A negative example is an instance is defined...

        Cosine similarity tells you how similar two instances are to each other.
        So we can get instances that are disimilar to each other. Then the negative
        instances positive tokens are the tokens we want to place the original instance under

        Sets the relevant instance variables

        :param float cutoff: percentage cutoff for negative scores
        """
        tagged_documents = []
        for instance, tokens in self.data.items():
            tagged_documents.append(TaggedDocument(tokens, [instance]))

        # Train doc2vec model for computing similarities between instances
        # negative set to 0 so no negative sampling will be used while training document model
        model = Doc2Vec(min_count=2, negative=0, workers=8)
        print('building vocab...')
        model.build_vocab(tagged_documents)
        print('training model...')
        model.train(tagged_documents, total_examples=model.corpus_count, epochs=10)
        print('done training model')

        instances_to_negative = {}

        for instance1 in self.pos_instances_to_tokens:
            tokens = []
            for instance2 in self.pos_instances_to_tokens:
                docvec1 = model.docvecs[instance1]
                docvec2 = model.docvecs[instance2]
                if cosine_similarity(docvec1, docvec2) <= cutoff:
                    #tokens += [t if t not in self.pos_tokens_to_instances[instance1] for t in self.pos_instances_to_tokens[instance2]]
                    for token in self.pos_instances_to_tokens[instance2]:
                        # If the token isn't already in the positive tokens of the instance, add the token.
                        # This is to avoid things like water_bottle and shampoo being negatives of each other,
                        # but bottle still shows up in the negative tokens of water_bottle because bottle shows up
                        # sometimes in the shampoo examples.
                        if token not in self.pos_instances_to_tokens[instance1]:
                            tokens.append(token)

            # add unique list of tokens
            instances_to_negative[instance1] = list(set(tokens))

        self._neg_instances_to_tokens = instances_to_negative

        # Invert the dict to map tokens -> instances
        neg_tokens_to_instances = defaultdict(list)
        for instance, tokens in self._neg_instances_to_tokens.items():
            for token in tokens:
                neg_tokens_to_instances[token].append(instance)

        self._neg_tokens_to_instances = dict(neg_tokens_to_instances)
