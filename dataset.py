import codecs
import math
import pickle
import re

import numpy as np
import pandas as pd
from pandas import read_table

from category import Category
from instance import Instance
from neg_sample_selection import NegSampleSelection
from token_ import Token

# whether to regen negative examples or load from previous file
argGenNeg = ARGS.negexmpl
resultDir = ARGS.resDir
NEG_SAMPLE_PORTION = float(ARGS.cutoff)

# This controls the minimum number of times a token has to appear in
# descriptions for an instance before the instance is deemed to be a positive
# example of this token
MIN_TOKEN_PER_INST = 2

def fileAppend(fName, sentence):
    """
    Function to write results/outputs to a log file
        Args: file descriptor, sentence to write
        Returns: Nothing
    """
    with open(fName, "a") as myfile:
        myfile.write(sentence)
        myfile.write("\n")

# get the number for the object label string
def get_object_num(object_str):
    match = re.search(r'\w+[^1-9][^1-9]_(\d.*)', object_str)
    return match.group(1)

class DataSet:
    """ Class to bundle data set related functions and variables """

    def __init__(self, path, anFile):
        """
        Initialization function for Dataset class
        Args:
            path - physical location of image dataset
            anFile - 6k amazon mechanical turk description file
        Returns: Nothing
        """
        self.dsPath = path
        self.annotationFile = anFile

    def findCategoryInstances(self):
        """
        Function to find all categories and instances in the dataset
        >> Read the amazon mechanical turk annotation file,
        >> Find all categories (ex, tomato), and instances (ex, tomato_1, tomato_2..)
        >> Create Category class instances and Instance class instances

        Args:  dataset instance
        Returns:  Category class instances, Instance class instances
        """
        nDf = read_table(self.annotationFile, sep=',', header=None)
        nDs = nDf.values
        categories = {}
        instances = {}
        for (k1, v1) in nDs:
            instName = k1.strip()
            (cat, inst) = instName.split("/")
            # issue with things like water_bottle
            num = get_object_num(inst)
            if cat not in categories.keys():
                categories[cat] = Category(cat)
            categories[cat].addCategoryInstances(num)
            if instName not in instances.keys():
                instances[instName] = Instance(instName, num)
        instDf = pd.DataFrame(instances, index=[0])
        catDf = pd.DataFrame(categories, index=[0])
        return (catDf, instDf)

    def splitTestInstances(self, cDf):
        """
        Function to find one instance from all categories for testing
        >> We use 4-fold cross validation here
        >> We try to find a random instance from all categories for testing

        Args:  dataset instance, all Category class instances
        Returns:  array of randomly selected instances for testing
        """
        cats = cDf.to_dict()
        tests = np.array([])
        for cat in np.sort(cats.keys()):
            obj = cats[cat]
            tests = np.append(tests, obj[0].chooseOneInstance())
        tests = np.sort(tests)
        return tests

    # PAT TODO: Check against repo for indentation
    def getDataSet(self, nDf, tests, fName):
        """
        Function to add amazon mechanical turk description file,
        find all tokens, find positive and negative instances for all tokens

        Args:  dataset instance, array of Category class instances,
            array of Instance class instances, array of instance names to test,
            file name for logging
        Returns:  array of Token class instances
        """
        instances = nDf.to_dict()
        # read the amazon mechanical turk description file line by line,
        # separating by comma [ line example, 'arch/arch_1, yellow arch'
        df = read_table(self.annotationFile, sep=',', header=None)
        tokenDf = {}
        # column[0] would be arch/arch_1 and column[1] would be 'yellow arch' """
        docs = {}
        for column in df.values:
            ds = column[0]
            if ds in docs.keys():
                sent = docs[ds]
                sent += " " + column[1]
                docs[ds] = sent
            else:
                docs[ds] = column[1]

        for inst in docs.keys():
            #get the counts for tokens and filter those < MIN_TOKEN_PER_INST

            token_counts = pd.Series(docs[inst].split(" ")).value_counts()
            token_counts = token_counts[token_counts >= MIN_TOKEN_PER_INST]
            dsTokens = token_counts.index.tolist()

            instances[inst][0].addTokens(dsTokens)
            for annotation in dsTokens:
                if annotation not in tokenDf.keys():
                    # creating Token class instances for all tokens (ex, 'yellow' and 'arch')
                    tokenDf[annotation] = Token(annotation)
                # add 'arch/arch_1' as a positive instance for token 'yellow'
                tokenDf[annotation].extendPositives(inst)

            #if ds not in tests:
            #iName = instances[ds][0].getName()
            #for annotation in dsTokens:
                #if annotation not in tokenDf.keys():
                    # creating Token class instances for all tokens (ex, 'yellow' and 'arch')
                    #tokenDf[annotation] = Token(annotation)
                # add 'arch/arch_1' as a positive instance for token 'yellow'
                #tokenDf[annotation].extendPositives(ds)
        tks = pd.DataFrame(tokenDf, index=[0])
        sent = "Tokens :: " + " ".join(tokenDf.keys())
        fileAppend(fName, sent)
        negSelection = NegSampleSelection(docs)

        # check if the negative examples
        if argGenNeg != "":
            with open(argGenNeg, 'rb') as handle:
                negExamples = pickle.load(handle)
        else:
            negExamples = negSelection.generateNegatives()

        # find negative instances for all tokens.
        token_negex = {}
        for tk in tokenDf.keys():
            poss = list(set(tokenDf[tk].getPositives()))
            # this keeps track of how strongly negative each instance is for this token
            negCandidateScores = {}

            for ds in poss:
                if isinstance(negExamples[ds], str):
                    negatives1 = negExamples[ds].split(",")
                    localNegCandScores = {}
                    for instNeg in negatives1:
                        s1 = instNeg.split("-")
                        #filter out instances that see the token in their descriptions
                        #also filter out instances that are in the test split
                        if s1[0] in tests or tk in docs[s1[0]].split(" "):
                            continue

                        localNegCandScores[s1[0]] = float(s1[1])
                        #sort the local dictionary by closeness and select the back 2/3 of that list
                    scores_sorted = list(sorted(localNegCandScores.items(), key=lambda x: x[1], reverse=False))
                    scores_sorted = scores_sorted[len(scores_sorted)//3:]
                    #now update the main dictionary
                    for (inst, val) in scores_sorted:
                        if inst in negCandidateScores:
                            negCandidateScores[inst] += val
                        else:
                            negCandidateScores[inst] = val


            # out of the options left choose the N most negative
            num_to_choose = int(math.ceil(float(len(negCandidateScores.keys())) * NEG_SAMPLE_PORTION))
            # TESTING: no more than twice as many negative examples as positive
            # num_to_choose = min(len(poss)*2,num_to_choose)
            choices_sorted = list(sorted(negCandidateScores.items(), key=lambda x: x[1], reverse=True))
            choices = [negInst for negInst, negVal in choices_sorted[:num_to_choose]]

            print(f"For token {tk} with {len(poss)} positive examples choosing {num_to_choose} examples out of {len(negCandidateScores.keys())}")
            # This dictionary is used to record the negative instances found for each token
            token_negex[tk] = list(set(choices))
            # print tk+":"+str(list(negCandidateScores.keys())).replace("[","").replace("]","")

            negsPart = choices
            tokenDf[tk].extendNegatives(negsPart)
            # print tk,":",token_negex[tk]

        all_tkns = list(token_negex.keys())
        max_insts = np.max([len(token_negex[key]) for key in token_negex])
        with codecs.open(resultDir + "/negative_insts.csv", "w", encoding="utf-8") as out_file:
            for tkn in all_tkns:
                tkn = tkn.encode("UTF-8")
                out_file.write(tkn)
                out_file.write(",")
            out_file.write("\n")
            for i in range(max_insts):
                for tkn in all_tkns:
                    if len(token_negex[tkn]) > i:
                        out_file.write(str(token_negex[tkn][i])+",")
                    else:
                        out_file.write(",")
                out_file.write("\n")

        return tks
