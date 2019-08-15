# -*- coding: utf-8 -*-
#!/usr/bin/env python
import argparse
import csv
import os
import re

from datetime import datetime

import numpy as np
from pandas import read_table
from sklearn import linear_model
from sklearn.pipeline import Pipeline

from dataset import DataSet

# this is a bit of a hack to get everything to default to utf-8
#importlib.reload(sys)
#sys.setdefaultencoding('UTF8')

# RAND_SEED = int(args.seed)
# random.seed(RAND_SEED)

# generalObjs = ['potatoe','cylinder','square', 'cuboid', 'sphere', 'halfcircle','circle','rectangle','cube','triangle','arch','semicircle','halfcylinder','wedge','block','apple','carrot','tomato','lemon','cherry','lime', 'banana','corn','hemisphere','cucumber','cabbage','ear','potato', 'plantain','eggplant']

# generalShapes = ['spherical', 'cylinder', 'square', 'rounded', 'cylindershaped', 'cuboid', 'rectangleshape','arcshape', 'sphere', 'archshaped', 'cubeshaped', 'curved' ,'rectangular', 'triangleshaped', 'halfcircle', 'globular','halfcylindrical', 'circle', 'rectangle', 'circular', 'cube', 'triangle', 'cubic', 'triangular', 'cylindrical','arch','semicircle', 'squareshape', 'arched','curve', 'halfcylinder', 'wedge', 'cylindershape', 'round', 'block', 'cuboidshaped']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resDir', required=True,
                        help='path to result directory')
    parser.add_argument('--cat', choices=['all', 'rgb', 'shape', 'object'], required=True,
                        help='type for learning')
    parser.add_argument('--pre', required=True,
                        help='the file with the preprocessed data')
    parser.add_argument('--cutoff', choices=['0.25', '0.5', '0.75'], default='0.25',
                        help='the cutoff for what portion of negative examples to use')
    parser.add_argument('--seed', default=None, required=False,
                        help='a random seed to use')
    parser.add_argument('--visfeat', default="ImgDz", required=False,
                        help='folder for features')
    parser.add_argument('--listof', default="list_of_instances.conf", required=False,
                        help='file for list of instances/files')
    parser.add_argument('--negexmpl', default="", required=False,
                        help='file where negative examples are saved')

    return parser.parse_known_args()

# get the object name from object label string
def get_object_name(object_str):
    match = re.search(r'(\w+[^1-9][^1-9])_\d+', object_str)
    return match.group(1)

# get the number for the object label string
def get_object_num(object_str):
    match = re.search(r'\w+[^1-9][^1-9]_(\d.*)', object_str)
    return match.group(1)

def fileAppend(fName, sentence):
    """
    Function to write results/outputs to a log file
        Args: file descriptor, sentence to write
        Returns: Nothing
    """
    with open(fName, "a") as myfile:
        myfile.write(sentence)
        myfile.write("\n")

def getTestFiles(insts, kind, tests, token):
    """
    Function to get all feature sets for testing and dummy 'Y' values
    Args:  Array of all Instance class instances, type of testing
        (rgb, shape, or object) , array of test instance names,
        token (word) that is testing
    Returns:  Feature set and values for testing
    """
    instances = insts.to_dict()
    features = []
    y = []
    for nInst in tests:
        y1 = instances[nInst][0].getY(token, kind)
        fs = instances[nInst][0].getFeatures(kind)
        features.append(list(fs))
        y.append(list(np.full(len(fs), y1)))
    return(features, y)

def getNonTestFiles(insts, kind, tests):
    """
    Function to get all feature sets for training data. This is used for
    testing on the training data (as a preliminary step to filter out tokens
    that are not meaningful like 'the')
    Args:  Array of all Instance class instances, type of testing
        (rgb, shape, or object) , array of test instance names,
        token (word) that is testing
    Returns:  Feature set
    """
    instances = insts.to_dict()
    features = []
    trainNames = []
    for nInst in instances.keys():
        if nInst not in tests:
            fs = instances[nInst][0].getFeatures(kind)
            trainNames.append(nInst)
            features.append(list(fs))

    return(features, trainNames)

def findTrainTestFeatures(insts, tkns, tests, kinds):
    """
    Function to iterate over all tokens, find train and test features for execution
    Args:  Array of all Instance class instances,
    array of all Token class instances,
    array of test instance names
    Returns:  all train test features, values, type of testing
    """
    tokenDict = tkns.to_dict()
    for token in np.sort(tokenDict.keys()):
    # for token in ['arch']:
        objTkn = tokenDict[token][0]
        for kind in kinds:
            # for kind in ['rgb']:
            (features, y) = objTkn.getTrainFiles(insts, kind)
            (testFeatures, testY) = getTestFiles(insts, kind, tests, token)
            (trainForTestingFeatures, trainNames) = getNonTestFiles(insts, kind, tests)
            if len(features) == 0:
                continue
            yield (token, kind, features, y, testFeatures, testY, trainForTestingFeatures, trainNames)

def callML(resultDir, insts, tkns, tests, kinds, fAnnotation):
    # generate a CSV result file with all probabilities
    # for the association between tokens (words) and test instances"""
    confFile = open(resultDir + '/groundTruthPrediction.csv', 'w')
    headFlag = 0
    fldNames = np.array(['Token', 'Type'])
    confWriter = csv.DictWriter(confFile, fieldnames=fldNames)

    # Generate another CSV file with the results of applying the classifiers back on
    # the training data. This is used for token filtering
    trainConfFile = open(resultDir + '/groundTruthPredictionTrain.csv', 'w')
    trainHeadFlag = False
    trainFldNames = set()

    # Trying to add correct object name of test instances in groundTruthPrediction csv file
    # ex, 'tomato/tomato_1 - red tomato'
    featureSet = read_table(fAnnotation, sep=',', header=None)
    featureSet = featureSet.values
    fSet = dict(zip(featureSet[:, 0], featureSet[:, 1]))
    testTokens = []

    # fine tokens, type to test, train/test features and values """
    for (token, kind, X, Y, tX, tY, trX, trNames) in findTrainTestFeatures(insts, tkns, tests, kinds):
        if token not in testTokens:
            testTokens.append(token)
    print(f"Token : {token} Kind : {kind}")
    # binary classifier Logisitc regression is used here
    sgdK = linear_model.LogisticRegression(C=10**5, random_state=0)
    pipeline2_2 = Pipeline([("logistic", sgdK)])

    pipeline2_2.fit(X, Y)
    fldNames = np.array(['Token', 'Type'])
    confD = {}
    confDict = {'Token': token, 'Type': kind}
    # testing all images category wise and saving the probabilitties in a Map
    # for ex, for category, tomato, test all images (tomato image 1, tomato image 2...)
    for ii in range(len(tX)):
        testX = tX[ii]

        tt = tests[ii]

        tProbs = []
        probK = pipeline2_2.predict_proba(testX)
        tProbs = probK[:, 1]

        for ik in range(len(tProbs)):
            fldNames = np.append(fldNames, str(ik) + "-" + tt)
            confD[str(ik) + "-" + tt] = str(fSet[tt])

        for ik in range(len(tProbs)):
            confDict[str(ik) + "-" + tt] = str(tProbs[ik])

    if headFlag == 0:
        headFlag = 1
        # saving the header of CSV file
        confWriter = csv.DictWriter(confFile, fieldnames=fldNames)
        confWriter.writeheader()

        confWriter.writerow(confD)
   # saving probabilities in CSV file
    confWriter.writerow(confDict)

    # now generate the probability of each training data being an example of this token and kind
    trainProbDict = {"Token": token, "Type": kind}
    trainConfD = {}

    for trI in range(len(trX)):
        trainIX = trX[trI]
        trName = trNames[trI]
        # print trName, trainIX

        probK = pipeline2_2.predict_proba(trainIX)
        tProbs = probK[:, 1]

        for ik in range(len(tProbs)):
            trainFldNames.add(str(ik) + "-" + trName)
            trainConfD[str(ik) + "-" + trName] = fSet[trName]

        for ik in range(len(tProbs)):
            trainProbDict[str(ik) + "-" + trName] = str(tProbs[ik])

    # should be unable to even predict the training data
    if not trainHeadFlag:
        trainHeadFlag = True
        print(f"writing the header len: {2+len(trainFldNames)}")
        # saving the header of CSV file
        trainConfWriter = csv.DictWriter(trainConfFile, fieldnames=['Token', 'Type'] + list(trainFldNames))
        trainConfWriter.writeheader()

        trainConfWriter.writerow(trainConfD)
        # saving probabilities in CSV file
        trainConfWriter.writerow(trainProbDict)


    confFile.close()
    trainConfFile.close()

def execution(resultDir, ds, cDf, nDf, tests, kinds, fAnnotation):
    os.mkdir(resultDir)
    os.mkdir(resultDir + "/NoOfDataPoints")
    resultDir1 = resultDir + "/NoOfDataPoints/6000"
    os.mkdir(resultDir1)

    fResName = resultDir1 + "/results.txt"
    sent = "Test Instances :: " + " ".join(tests)
    fileAppend(fResName, sent)
    # read amazon mechanical turk file, find all tokens
    # get positive and negative instance for all tokens
    tokens = ds.getDataSet(cDf, nDf, tests, fResName)
    # Train and run binary classifiers for all tokens, find the probabilities
    # for the associations between all tokens and test instances,
    # and log the probabilitties
    callML(resultDir1, nDf, tokens, tests, kinds, fAnnotation)

def main():
    ARGS, unused = parse_args()

    resultDir = ARGS.resDir

    preFile = ARGS.pre
    kinds = np.array([ARGS.cat])
    if ARGS.cat == 'all':
        kinds = np.array(['rgb', 'shape', 'object'])

    execPath = './'
    dPath = "../"
    dsPath = dPath + ARGS.visfeat
    fAnnotation = execPath + ARGS.listof

    ds = ""
    cDf = ""
    nDf = ""
    tests = ""

    print("START :: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    anFile = execPath + preFile
    #os.system("mkdir -p " + resultDir)
    # creating a Dataset class Instance with dataset path, amazon mechanical turk description file
    ds = DataSet(dsPath, anFile)
    # find all categories and instances in the dataset
    (cDf, nDf) = ds.findCategoryInstances()

    # find all test instances. We are doing 4- fold cross validation
    tests = ds.splitTestInstances(cDf)
    print(f"ML START :: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    execution(resultDir, ds, cDf, nDf, tests, kinds, fAnnotation)
    print("ML END :: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
