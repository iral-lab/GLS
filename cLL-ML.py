#!/usr/bin/env python
import numpy as np
import numpy.linalg as la
from itertools import product
from numpy import linalg as LA
from scipy.linalg import orth
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
from pandas import DataFrame, read_table
import pandas as pd
from scipy.special import logsumexp
import random
import scipy.stats
import collections
from sklearn import mixture
import random
from collections import Counter
import json
import os
import math
import sys
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,f1_score)
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import csv 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import sklearn
import argparse
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from scipy import spatial
from sklearn.utils.extmath import softmax

parser = argparse.ArgumentParser()
parser.add_argument('--resDir',help='path to result directory',required=True)
parser.add_argument('--noDPs', help='number of data points. Maximum is 4060',required=True,type=int)
parser.add_argument('--tp',  help='testing instance number to pick',choices=range(4),default=0,type=int)
parser.add_argument('--cat', help='category for testing', choices=['rgb','shape','object'],required=True)
parser.add_argument('--pre', help='the file with the preprocessed data', required=True)
parser.add_argument('--visfeat',help='folder for features', default="ImgDz", required=False)
subparsers = parser.add_subparsers(help='projects')

al_parser = subparsers.add_parser('al', help='Active Learning Project')
al_parser.add_argument('--execType', help='execution type', choices=['alpool','aluncert','aldpp','algmmdpp'], required=True)
al_parser.add_argument('--gmmPts', help='GMM Number of Components',default=10,type=int)
al_parser.add_argument('--generate', help='ReLearn GMM clusters', default=0,choices=[0,1],type=int)
al_parser.add_argument('--iteration', help='Iteration', default=1,choices=[1,2,3,4],type=int)
al_parser.add_argument('--selectionType', help='Data Point Selection type', default='max', choices=['max','entropy'])
al_parser.add_argument('--dppPts', help='K of DPP',default=5,type=int)

ml_parser = subparsers.add_parser('ml', help='Random Labeling Project')
ml_parser.add_argument('--execType', help='execution type', choices=['seq','random'], required=True)

args = parser.parse_args()

resultDir = args.resDir
numberOfDPs = args.noDPs
testInstanceID = args.tp
ctestInstanceID = testInstanceID
kinds = np.array([args.cat])
execType = args.execType
gmmPoints = 10
gmmGenerate = 0
iteractionNo = 1
selectionType = 'max'
dppPts = 5

preFile = args.pre

#execPath = '/Users/nishapillai/Documents/GitHub/alExec/'
execPath = './'
dPath = "../"
dsPath = dPath + args.visfeat
#dsPath = dPath + "ImgDz/"
fAnnotation = execPath + "groundtruth_annotation.conf"

sections = 5
quartiles = [2]

dgAbove = 80

ds = ""
cDf = ""
nDf = "" 
tests = ""

generalColors = ['yellow','blue','purple','black','isyellow','green','brown','orange','white','red']

generalObjs = ['potatoe','cylinder','square', 'cuboid', 'sphere', 'halfcircle','circle','rectangle','cube','triangle','arch','semicircle','halfcylinder','wedge','block','apple','carrot','tomato','lemon','cherry','lime', 'banana','corn','hemisphere','cucumber','cabbage','ear','potato', 'plantain','eggplant']

generalShapes = ['spherical', 'cylinder', 'square', 'rounded', 'cylindershaped', 'cuboid', 'rectangleshape','arcshape', 'sphere', 'archshaped', 'cubeshaped', 'curved' ,'rectangular', 'triangleshaped', 'halfcircle', 'globular','halfcylindrical', 'circle', 'rectangle', 'circular', 'cube', 'triangle', 'cubic', 'triangular', 'cylindrical','arch','semicircle', 'squareshape', 'arched','curve', 'halfcylinder', 'wedge', 'cylindershape', 'round', 'block', 'cuboidshaped']


def fileAppend(fName, sentence):
  """""""""""""""""""""""""""""""""""""""""
	Function to write results/outputs to a log file
	 	Args: file descriptor, sentence to write
	 	Returns: Nothing
  """""""""""""""""""""""""""""""""""""""""
  with open(fName, "a") as myfile:
    myfile.write(sentence)
    myfile.write("\n")

######## Negative Example Generation##########
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
    
    def sentences_perm(self):
        from random import shuffle
        shuffle(self.sentences)
        return self.sentences

class NegSampleSelection:
   """ Class to bundle negative example generation functions and variables. """
   __slots__ = ['docs']
   docs = {}
   def __init__(self,docs):
      """""""""""""""""""""""""""""""""""""""""
                Initialization function for NegSampleSelection class
                Args: Documents dictionary where key is object instance and value 
                      is object annotation
                Returns: Nothing
      """""""""""""""""""""""""""""""""""""""""
      docs = collections.OrderedDict(sorted(docs.items()))
      self.docs = docs

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
   
   def square_rooted(self,x):
      return round(math.sqrt(sum([a*a for a in x])),3)
 
   def cosine_similarity(self,x,y):
      numerator = sum(a*b for a,b in zip(x,y))
      denominator = self.square_rooted(x)*self.square_rooted(y)
      return round(numerator/float(denominator),3)

   def generateNegatives(self):
      docs = self.docs
      docNames = docs.keys()
      docLists = self.sentenceToWordLists()
      docDicts = self.sentenceToWordDicts()
      docLabels = []
      for key in docNames:
        ar = key.split("/")
        docLabels.append(ar[1])
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
      for i , item1 in enumerate(docLabels):
         fDoc = model.docvecs[docLabels[i]]
         cInstMap = {}
         cInstance = docNames[i]
         for j,item2 in enumerate(docLabels):
           tInstance = docNames[j]
           if item1 == item2:
            cInstMap[tInstance] = 0.0
           else:
            tDoc = model.docvecs[docLabels[j]]
            cosineVal = self.cosine_similarity(fDoc,tDoc)
            cValue = math.degrees(math.acos(cosineVal))
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
      return negInstances

############Negative Example Generation --- END ########

###################GMM class starts here#################
class GMM:
   def __init__(self,noComps,data):
      self.noComps = noComps
      self.arMean = []
      self.arCovs = []
      
      lnData = len(data) / noComps
      for i in range(noComps):
          nData = None
          if i == (noComps - 1):
              nData = data[i * lnData:]
          else:
              nData = data[i * lnData:(i + 1) * lnData]
          self.arMean.append(np.mean(nData, axis=0))
          nCovariance = np.cov(nData, rowvar=0)
#          det = np.fabs(la.det(nCovariance))
#          factor = (2.0 * np.pi)**(data.shape[1] / 2.0) * (det)**(0.5)
          self.arCovs.append(nCovariance)

      self.priors = np.ones(noComps, dtype = "double") / noComps

   def pdf(self,i, x):

        print "Co variance ", i , " ->",self.arCovs[i]

        dx = x - self.arMean[i]
        A =  la.inv(np.array(self.arCovs[i]))
        print la.det(self.arCovs[i])
        fE = (la.det(self.arCovs[i]) * 2.0 * np.pi) ** 0.5 
        gS = np.exp(-0.5 * np.dot(np.dot(dx,A),dx)) / fE
        return gS

   def em(self, data, nsteps = 10):

        k = self.noComps
        d = data.shape[1]
        n = len(data)

        for l in range(nsteps):
            print "STEPS", l
            # E step

            responses = np.zeros((k,n))
            for j in range(n):
                for i in range(k):
                    pdfVal = self.pdf(i,data[j])
                    responses[i,j] = self.priors[i] * pdfVal
            print "RESPONSES START",responses.shape
            print responses
            responses = responses / np.sum(responses,axis=0) # normalize the weights
            print "2nd Normalized RESPONSESS",responses
            print "Responses -- DONE"
            # M step
            N = np.sum(responses,axis=1)
            print "N",N, N.shape
            for i in range(k):
                print "I ", i , " in K ", k
                mu = np.dot(responses[i,:],data) / N[i]
                print "MU ",mu
                sigma = np.zeros((d,d))
                print "SPECIAL", d
                for j in range(n):
#                   sigma += responses[i,j] * np.outer(data[j,:] - mu, data[j,:] - mu)
#                   sigma += responses[i,j] * np.dot(data[j,:] - mu, data[j,:] - mu)
                   ch = np.array(data[j,:] - mu)
                   ch = np.reshape(ch,(ch.shape[0],1))
                   sigma += responses[i,j] * np.dot(ch,ch.T)
                   print data[j,:] - mu
                   print np.dot(ch,ch.T)
                   print responses[i,j]
                   print "ADDING COMPONENT",responses[i,j] * np.dot(ch,ch.T)

                   print "NEW SIGMA FOR ",j,"-->",sigma
                print "SIGMA ", sigma
                sigma = sigma / N[i]
                print "SIGMA NORM ", sigma
                self.arMean[i] = mu
                self.arCovs[i] = sigma
#                det = np.fabs(la.det(sigma))
#                print "DETER ", det
#                factor = (2.0 * np.pi)**(data.shape[1] / 2.0) * (det)**(0.5)
#                print "FACTOR ", factor
#                self.arFactors[i] = factor
                self.priors[i] = N[i] / np.sum(N) # normalize the new priors
                print "NEW PRIORS ", self.priors
        print "THIS STEP IS FINISHED"
  
###########################################################

class Category:
   """ Class to bundle our dataset functions and variables category wise. """
   __slots__ = ['catNums', 'name']  
   catNums = np.array([], dtype='object')
  
   def __init__(self, name):
      """""""""""""""""""""""""""""""""""""""""
		Initialization function for category class
     		Args: category name
     		Returns: Nothing
      """""""""""""""""""""""""""""""""""""""""     	
      self.name = name
      
   def getName(self):
      """""""""""""""""""""""""""""""""""""""""
      	Function to get the category name
     		Args: category class instance
     		Returns: category name
      """"""""""""""""""""""""""""""""""""""""" 	   
      return self.name
     
   def addCategoryInstances(self,*num):
      """""""""""""""""""""""""""""""""""""""""
      Function to add a new instance number to the category
         Args: category class instance
         Returns: None
      """""""""""""""""""""""""""""""""""""""""  	   
      self.catNums = np.unique(np.append(self.catNums,num))


   def chooseOneInstance(self):
      """""""""""""""""""""""""""""""""""""""""
      Function to select one random instance from this category for testing
         Args: category class instance
         Returns: Randomly selected instance name
      """""""""""""""""""""""""""""""""""""""""    	   
      r = random.randint(0,self.catNums.size - 1)  
# 	  r = testInstanceID
      instName = self.name + "/" + self.name + "_" + self.catNums[r]
      return instName


class Instance(Category):
	""" Class to bundle instance wise functions and variables """
	__slots__ = ['name','catNum','tokens','negs','gT']
	gT = {}
	tokens = np.array([])
	name = ''
	def __init__(self, name,num):
		"""""""""""""""""""""""""""""""""""""""""
		Initialization function for Instance class
         Args: instance name, category number of this instance
         Returns: Nothing
        """""""""""""""""""""""""""""""""""""""""     	
		self.name = name
		self.catNum = num

	def getName(self):
		"""""""""""""""""""""""""""""""""""""""""

    	Function to get the instance name
     		Args: Instance class instance
     		Returns: instance name
     	"""""""""""""""""""""""""""""""""""""""""
		return self.name     	
        
   	
        
	def getFeatures(self,kind):
		"""""""""""""""""""""""""""""""""""""""""
		Function to find the complete dataset file path (.../arch/arch_1/arch_1_rgb.log) 
		where the visual feaures are stored, read the features from the file, and return
	
         	Args: Instance class instance, type of features(rgb, shape, or object)
         	Returns: feature set
        """""""""""""""""""""""""""""""""""""""""     	
		instName = self.name
		instName.strip()
		ar1 = instName.split("/")
		path1 = "/".join([dsPath,instName])
		path  = path1 + "/" + ar1[1] + "_" + kind + ".log"
		featureSet = read_table(path,sep=',',  header=None)
		return featureSet.values        
        
	def addNegatives(self, negs):
		"""""""""""""""""""""""""""""""""""""""""
		Function to add negative instances
	
         	Args: Instance class instance, array of negative instances 
         	Returns: None
        """""""""""""""""""""""""""""""""""""""""     	
		add = lambda x : np.unique(map(str.strip,x))
		self.negs = add(negs)

	def getNegatives(self):
		"""""""""""""""""""""""""""""""""""""""""
		Function to get the list of negative instances
	
         	Args: Instance class instance 
         	Returns: array of negative instances
        """""""""""""""""""""""""""""""""""""""""    	
		return self.negs

	def addTokens(self,tkn):
		"""""""""""""""""""""""""""""""""""""""""
		Function to add a word (token) describing this instance to the array of tokens
         Args: Instance class instance, word
         Returns: None
        """""""""""""""""""""""""""""""""""""""""       	
		self.tokens = np.append(self.tokens,tkn)

	def getTokens(self):
		"""""""""""""""""""""""""""""""""""""""""
		Function to get array of tokens which humans used to describe this instance
         Args: Instance class instance
         Returns: array of words (tokens)
        """""""""""""""""""""""""""""""""""""""""       	
		return self.tokens
  
	def getY(self,token,kind):		
		"""""""""""""""""""""""""""""""""""""""""
        Function to find if a token is a meaningful representation for this instance for testing. In other words, if the token is described for this instance in learning phase, we consider it as a meaningful label.
         Args: Instance class instance, word (token) to verify, type of testing
         Returns: 1 (the token is a meaningful label) / 0 (the token is not a  meaningful label)
		""""""""""""""""""""""""""""""""""""""""" 		
		if token in list(self.tokens):
			if kind == "rgb":
				if token in list(generalColors):
					return 1
			elif kind == "shape":
				if token in list(generalShapes):
					return 1
			else:
				if token in list(generalObjs):
					return 1
		return 0

class Token:
	
   """ Class to bundle token (word) related functions and variables """
   __slots__ = ['name', 'posInstances', 'negInstances']
   posInstances = np.array([], dtype='object')
   negInstances = np.array([], dtype='object')

   def __init__(self, name):
        """""""""""""""""""""""""""""""""""""""""
		Initialization function for Token class
         Args: token name ("red")
         Returns: Nothing
        """""""""""""""""""""""""""""""""""""""""    	   
        self.name = name
   
   def getTokenName(self):
       """""""""""""""""""""""""""""""""""""""""
    	Function to get the label from class instance
     		Args: Token class instance
     		Returns: token (label, for ex: "red")
       """""""""""""""""""""""""""""""""""""""""   	   
       return self.name

   def extendPositives(self,instName):
      """""""""""""""""""""""""""""""""""""""""
		Function to add postive instance (tomato/tomato_1) for this token (red)
	
         	Args: token class instance, positive instance
         	Returns: None
      """""""""""""""""""""""""""""""""""""""""      	   
      self.posInstances = np.append(self.posInstances,instName)
   
   def getPositives(self):
      """""""""""""""""""""""""""""""""""""""""
		Function to get all postive instances of this token 
	
         	Args: token class instance
         	Returns: array of positive instances (ex: tomato/tomato_1, ..)
      """""""""""""""""""""""""""""""""""""""""    	   
      return self.posInstances

   def extendNegatives(self,*instName):
      """""""""""""""""""""""""""""""""""""""""
		Function to add negative instances for this token
	
         	Args: Instance class instance, array of negative instances 
         	Returns: None
      """""""""""""""""""""""""""""""""""""""""    	   
      self.negInstances = np.unique(np.append(self.negInstances,instName))

   def getNegatives(self):
      """""""""""""""""""""""""""""""""""""""""
		Function to get all negative instances of this token (ex, "red")
	
         	Args: token class instance
         	Returns: array of negative instances (ex: arch/arch_1, ..)
      """"""""""""""""""""""""""""""""""""""""" 	
      return self.negInstances

   def clearNegatives(self):
      self.negInstances = np.array([])

   def shuffle(self,a, b, rand_state):
      rand_state.shuffle(a)
      rand_state.shuffle(b)

   def getTrainFiles(self,insts,kind):
      """""""""""""""""""""""""""""""""""""""""
        This function is to get all training features for this particular token
    		>> Find positive instances described for this token
    		>> if the token is used less than 3 times, remove it from execution
    		>> fetch the feature values from the physical dataset location
    		>> find negative instances and fetch the feature values from the physical location
    		>> balance the number positive and negative feature samples
		
             Args: token class instance, complete Instance list, type for learning/testing
             Returns: training features (X) and values (Y)
      """""""""""""""""""""""""""""""""""""""""    	   
      instances = insts.to_dict()
      pS = Counter(self.posInstances)
      if len(self.posInstances) <= 3:
      	  return np.array([]),np.array([])
      features = np.array([])
      negFeatures = np.array([])
      y = np.array([])
      if self.posInstances.shape[0] == 0 or self.negInstances.shape[0] == 0 :
         return (features,y)
      if self.posInstances.shape[0] > 0 :
        features = np.vstack(instances[inst][0].getFeatures(kind) for inst in self.posInstances)
      if self.negInstances.shape[0] > 0:
        negFeatures = np.vstack(instances[inst][0].getFeatures(kind) for inst in self.negInstances if len(inst) > 1)
        """ if length of positive samples are more than the length of negative samples, 
        duplicate negative instances to balance the count"""
        if len(features) > len(negFeatures):
          c = int(len(features) / len(negFeatures))
          negFeatures = np.tile(negFeatures,(c,1))
      if self.posInstances.shape[0] > 0 and self.negInstances.shape[0] > 0 :
       """ if length of positive samples are less than the length of negative samples, 
        duplicate positive samples to balance the count"""      	  
       if len(negFeatures) > len(features):
          c = int(len(negFeatures) / len(features))
          features = np.tile(features,(c,1))
      """ find trainY for our binary classifier: 1 for positive samples, 
      0 for negative samples"""          
      y = np.concatenate((np.full(len(features),1),np.full(len(negFeatures),0)))
      if self.negInstances.shape[0] > 0:
        features = np.vstack([features,negFeatures])
      return(features,y)


class DataSet:
   """ Class to bundle data set related functions and variables """	  
   __slots__ = ['dsPath', 'annotationFile']
   
   def __init__(self, path,anFile):
      """""""""""""""""""""""""""""""""""""""""
		Initialization function for Dataset class
         Args: 
         	path - physical location of image dataset
         	anFile - 6k amazon mechanical turk description file
         Returns: Nothing
      """""""""""""""""""""""""""""""""""""""""   	   
      self.dsPath = path
      self.annotationFile = anFile

   def findCategoryInstances(self):
      """""""""""""""""""""""""""""""""""""""""
        Function to find all categories and instances in the dataset
        	>> Read the amazon mechanical turk annotation file,
        	>> Find all categories (ex, tomato), and instances (ex, tomato_1, tomato_2..)
        	>> Create Category class instances and Instance class instances 

             Args:  dataset instance
             Returns:  Category class instances, Instance class instances
      """""""""""""""""""""""""""""""""""""""""     	   
      nDf = read_table(self.annotationFile,sep=',',  header=None)
      nDs = nDf.values
      categories = {}
      instances = {}
      for (k1,v1) in nDs:
          instName = k1.strip()
          (cat,inst) = instName.split("/")
          (_,num) = inst.split("_")
          if cat not in categories.keys():
             categories[cat] = Category(cat)
          categories[cat].addCategoryInstances(num)
          if instName not in instances.keys():
             instances[instName] = Instance(instName,num)
      instDf = pd.DataFrame(instances,index=[0])
      catDf =  pd.DataFrame(categories,index=[0])
      return (catDf,instDf)


   def splitTestInstances(self,cDf):
      """""""""""""""""""""""""""""""""""""""""
        Function to find one instance from all categories for testing
        	>> We use 4-fold cross validation here
        	>> We try to find a random instance from all categories for testing

             Args:  dataset instance, all Category class instances
             Returns:  array of randomly selected instances for testing
      """""""""""""""""""""""""""""""""""""""""     	   
      cats = cDf.to_dict()
      tests = np.array([])
      for cat in np.sort(cats.keys()):
         obj = cats[cat]
         tests = np.append(tests,obj[0].chooseOneInstance())
      tests = np.sort(tests)
      return tests

############################K-DPP implementation#######################

#@Courtesy :::  https://github.com/mbp28/determinantal-point-processes
# @C matlab scripts from http://www.alexkulesza.com/


   def elem_sympoly(self,vals, k):
    """Uses Newton's identities to compute elementary symmetric polynomials."""
    N = vals.shape[0]
    E = np.zeros([k+1, N+1])
    E[0,] = 1

    for i in range(1, k+1):
        for j in range(1, N+1):
            E[i,j] = E[i, j-1] + vals[j-1] * E[i-1, j-1]

    return E

   def sample_k(self,vals, k):
      N = vals.shape[0]
      E = self.elem_sympoly(vals, k)
      sample = np.zeros(k, dtype=int)
      rem = k
      while rem > 0:
        if N == rem:
         marg = 1
        else:
         marg = vals[N - 1] * E[rem - 1 ,N - 1]/ E[rem,N]
        if np.random.rand() < marg:
          sample[rem - 1] = N - 1
          rem = rem - 1
        N = N - 1
      return sample

   def sample_dpp(self,vals, vecs, k=0, one_hot=False):
    """
    This function expects

    Arguments:
    vals: NumPy 1D Array of Eigenvalues of Kernel Matrix
    vecs: Numpy 2D Array of Eigenvectors of Kernel Matrix
    """
    n = vecs.shape[0]
    # k-DPP
    if k:
        index = self.sample_k(vals, k) # sample_k, need to return index      
    if k == n:
      return np.arange(k)
    V = vecs[:, index]
    # Sample a set of k items
    items = list()
    for i in range(k):
     p = np.sum(V**2, axis=1)
     if np.sum(p) != 0:
      p = np.cumsum(p / np.sum(p)) # item cumulative probabilities
      item = (np.random.rand() <= p).argmax()
      items.append(item)
     # Delete one eigenvector not orthogonal to e_item and find new basis
      j = (np.abs(V[item, :]) > 0).argmax()
      Vj = V[:, j]
      V = orth(V - (np.outer(Vj,(V[item, :] / Vj[item]))))
    items.sort()
    sample = np.array(items)
    return sample


   def decompose_kernel(self,M):
      (V,D) = LA.eig(M);
      V = np.real(V);
      return (D,V)

   def getPointsDPP(self, data,k):
      min_max_scaler = preprocessing.MinMaxScaler()
      z = min_max_scaler.fit_transform(data)
      z = data
      s = 0.1
      sqdists = squareform(pdist(z, 'sqeuclidean'))
      L = scipy.exp(-sqdists / s**2)
      if execType == 'algmmdpp':
          gmm = mixture.GaussianMixture(n_components=gmmPoints, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)
          gmm.fit(data)
          densityProbs = gmm.predict_proba(data)         
          



          probIndEntropy = []
          count = 0
          for k1 in range(len(densityProbs)):
           probInds = [float('%.5f' % prob) for prob in densityProbs[k1]]
           chk = 0
           for prob in probInds:
              if prob != 0.0 and prob != 1.0:
               chk = 1
           if chk == 1:
              count += 1   
           probInds = [float(0.00000001) if prob == 0.0 else float(prob) for prob in probInds]
           probInds = np.true_divide(probInds,sum(probInds))
           entropyPdfs = [-1 * prob * np.log(prob) for prob in probInds]
           probIndEntropy.append(sum(entropyPdfs))
#          print densityProbs
          gmmPointsAr = np.zeros((len(densityProbs),len(densityProbs)))
          for i in range(len(densityProbs)):
            for j in range(len(densityProbs)):
              gmmPointsAr[i,j] = probIndEntropy[i]
           
          gmmPointsAr = np.array(gmmPointsAr)
          L = np.matmul(gmmPointsAr,L)
          L = np.matmul(L,gmmPointsAr)
          print "Non zero instances ",count
      qq = pdist(z, 'sqeuclidean')
      vecs,vals = self.decompose_kernel(L)
      dppIndices = self.sample_dpp(vals, vecs, k)
      return dppIndices
#######################################################################
   
   def generateALDPPTrainingInstances(self,nDf,tests,kind,totalNeeded):
      df = read_table(self.annotationFile, sep=',',  header=None)
      cDz = df.values
      instances = nDf.to_dict()
      if totalNeeded >= len(cDz):
         return range(len(cDz))
      instIndices = {}
      instFiles = []
#      print cDz
      for column in cDz:
          instFiles.append(column[0])

      for instName in instances.keys():

         inds = np.argwhere(np.array(instFiles)==instName)
         ars = [ind[0] for ind in inds]
         instIndices[instName] = ars
#      data = [list(inst) for instName in instances.keys() if instName not in tests for inst in instances[instName][0].getFeatures(kind)]

      data = []
      dataInsts = []
      for instName in instances.keys():
       if instName not in tests:
           instsData = instances[instName][0].getFeatures(kind)
           data.append(instsData[0])
           dataInsts.append(instName)
#      dataInsts = [instName for instName in instances.keys() if instName not in tests for inst in instances[instName][0].getFeatures(kind)]
#      dataInsts = [column[0] for idx,column in enumerate(cDz) if column[0] not in tests]
      trainInstIndices = sum([len( instIndices[instName]) for instName in instances.keys() if instName not in tests])
      data = np.array(data)
      relPoints = len(dataInsts)
      relDataSets = []
      dataInsts1 = dataInsts
      data1 = data
      if relPoints > 0:
        while relPoints > len(relDataSets):
           k = relPoints - len(relDataSets)
           if k > dppPts:
             k = dppPts
           dppIndices = []
           while k != len(dppIndices):
                 dppIndices = self.getPointsDPP(data1,k)
           relDataSets.extend([dataInsts1[i] for i in dppIndices])
           dataInsts1 = [dataInsts1[c] for c in range(len(dataInsts1)) if c not in dppIndices]
           data1 = np.array([data1[c] for c in range(len(data1)) if c not in dppIndices])
      print len(relDataSets),len(list(set(relDataSets)))
      indicesTobeTrained = []
      while(trainInstIndices > len(indicesTobeTrained)):
        for inst in relDataSets:
          if len(instIndices[inst]) > 0:
               ars = instIndices[inst]
               indicesTobeTrained.append(ars.pop(0))
               instIndices[inst] = ars
      strIndicesToBeTrained = [str(ind) for ind in indicesTobeTrained]
      lineIndices = " ".join(strIndicesToBeTrained)
      os.system("echo \'" + lineIndices + "\' >  GMMIndices/indicesGMMbased.log")
      print len(indicesTobeTrained),len(list(set(indicesTobeTrained)))
      if totalNeeded > len(indicesTobeTrained):
             return list(np.sort(indicesTobeTrained))
      return list(np.sort(indicesTobeTrained[0:totalNeeded]))

   def getALDPPTrainingInstances(self,nDf,tests,kind,totalNeeded):
      indicesFile = "GMMIndices/" + str(iteractionNo) + "-" + str(testInstanceID) + "-" + str(execType) + "-" + str(kind) + "-indicesGMMbased.log"
      df = read_table(indicesFile, sep=' ',  header=None)
      cDz = df.values
      indicesTobeTrained = [int(ind) for ind in cDz[0]]
      if totalNeeded > len(indicesTobeTrained):
             return list(np.sort(indicesTobeTrained))
      return list(indicesTobeTrained[0:totalNeeded])
#      return list(np.sort(indicesTobeTrained[0:totalNeeded]))

   def getALDPPTrainingInstances1(self,nDf,tests,kind,totalNeeded):
      global noComponents
#      print "Acquiring AL testcase indices"
      df = read_table(self.annotationFile, sep=',',  header=None)
      cDz = df.values
      instances = nDf.to_dict()
      if totalNeeded >= len(cDz):
         return range(len(cDz))
      instIndices = {}
      instFiles = []
#      print cDz
      for column in cDz:
          instFiles.append(column[0])

      for instName in instances.keys():

         inds = np.argwhere(np.array(instFiles)==instName)
         ars = [ind[0] for ind in inds]
         instIndices[instName] = ars

      data = [list(inst) for instName in instances.keys() if instName not in tests for inst in instances[instName][0].getFeatures(kind)]

      dataInsts = [instName for instName in instances.keys() if instName not in tests for inst in instances[instName][0].getFeatures(kind)]

      trainInstIndices = sum([len( instIndices[instName]) for instName in instances.keys() if instName not in tests])
 
      data = np.array(data)
      relPoints = trainInstIndices
      relDataSets = []
      if relPoints >= len(dataInsts):
         relPoints = len(dataInsts)
#        relDataSets = [instFiles[i % len(instFiles)] for i in range(relPoints)]
#      else:

      if relPoints > 0:
              k = 0
              if totalNeeded > len(dataInsts):
                     k = len(dataInsts)
              else:
                     k = totalNeeded
              dppIndices = []
              while k != len(dppIndices):
                 dppIndices = self.getPointsDPP(data,k)
              relDataSets.extend([dataInsts[i] for i in dppIndices])
      if totalNeeded > trainInstIndices:
          totalNeeded = trainInstIndices
      availIndices = sum([len( instIndices[instName]) for instName in list(set(relDataSets))])
      indicesTobeTrained = []
      count = 0
      while(availIndices > len(indicesTobeTrained)):
          inst = relDataSets[count % len(relDataSets)]
          count = count + 1
          if len(instIndices[inst]) > 0:
               ars = instIndices[inst]
               indicesTobeTrained.append(ars.pop(0))
               instIndices[inst] = ars
      remInstances = list(set(instances.keys()) - set(relDataSets) - set(tests))
      while(trainInstIndices > len(indicesTobeTrained)):
          for inst in remInstances:
            if len(instIndices[inst]) > 0 :
               ars = instIndices[inst]
               indicesTobeTrained.append(ars.pop(0))
               instIndices[inst] = ars
      return list(np.sort(indicesTobeTrained[0:totalNeeded]))


   def getALTrainingInstancesPool2(self,nDf,tests,kind,totalNeeded):
      global noComponents
#      print "Acquiring AL testcase indices"
      df = read_table(self.annotationFile, sep=',',  header=None)
      cDz = df.values
      instances = nDf.to_dict()
      if totalNeeded >= len(cDz):
         return range(len(cDz))
      instIndices = {}
      instFiles = []
#      print cDz
      for column in cDz:
          instFiles.append(column[0])

      for instName in instances.keys():

         inds = np.argwhere(np.array(instFiles)==instName)
         ars = [ind[0] for ind in inds]
         instIndices[instName] = ars

#      data = [list(inst) for idx,column in enumerate(cDz) if column[0] not in tests for inst in instances[column[0]][0].getFeatures(kind)]

#      data = [list(inst) for instName in instances.keys() if instName not in tests for inst in instances[instName][0].getFeatures(kind)]
      data = []
      dataInsts = []
      for idx,column in enumerate(cDz):
        if column[0] not in tests:
           instsData = instances[column[0]][0].getFeatures(kind)
           data.append(instsData[0])
           dataInsts.append(column[0])

#      dataInsts = [column[0] for idx,column in enumerate(cDz) if column[0] not in tests for inst in instances[column[0]][0].getFeatures(kind)]
#      dataInsts = [instName for instName in instances.keys() if instName not in tests for inst in instances[instName][0].getFeatures(kind)]
#      trainInstIndices = sum([len( instIndices[instName]) for instName in instances.keys() if instName not in tests])
      trainInstIndices = len(dataInsts)
      data = np.array(data)
      relPoints = trainInstIndices
      relDataSets = []
      if relPoints >= len(dataInsts):
         relPoints = len(dataInsts)
#        relDataSets = [instFiles[i % len(instFiles)] for i in range(relPoints)]
#      else:
      if relPoints > 0:
          gmm = mixture.GaussianMixture(n_components=25, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)
          gmm.fit(data)
          centers = []
          for i in range(gmm.n_components):
            density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(data)
            cmptNeeded = relPoints / gmm.n_components
            if i < (relPoints % gmm.n_components):
                cmptNeeded += 1
            dens1 = np.argsort(density)
            dens1 = np.flipud(dens1)
            den1 = dens1[0:cmptNeeded]
            centers.append(den1)
            relDataSets.extend([dataInsts[i] for i in den1])
      if totalNeeded > trainInstIndices:
          totalNeeded = trainInstIndices
      availIndices = sum([len( instIndices[instName]) for instName in list(set(relDataSets))])
      indicesTobeTrained = []
      while(availIndices > len(indicesTobeTrained)):
          for inst in relDataSets:
            if len(instIndices[inst]) > 0 :
               ars = instIndices[inst]
               indicesTobeTrained.append(ars.pop(0))
               instIndices[inst] = ars
      remInstances = list(set(instances.keys()) - set(relDataSets) - set(tests))
      while(trainInstIndices > len(indicesTobeTrained)):
          for inst in remInstances:
            if len(instIndices[inst]) > 0 :
               ars = instIndices[inst]
               indicesTobeTrained.append(ars.pop(0))
               instIndices[inst] = ars
      return list(np.sort(indicesTobeTrained[0:totalNeeded]))

   def getALTrainingInstancesPool(self,nDf,tests,kind,totalNeeded):
      df = read_table(self.annotationFile, sep=',',  header=None)
      cDz = df.values
      instances = nDf.to_dict()
      if totalNeeded >= len(cDz):
         return range(len(cDz))
      indicesFile = "GMMIndices/" + str(iteractionNo) + "-" + str(testInstanceID) + "-" + str(execType) + "-" + str(kind) + "-indicesGMMbased.log"
      df = read_table(indicesFile, sep=' ',  header=None)
      cDz = df.values
      indicesTobeTrained = [int(ind) for ind in cDz[0]]
      if totalNeeded > len(indicesTobeTrained):
             return list(np.sort(indicesTobeTrained))
      if execType == 'alpool':
            return list(np.sort(indicesTobeTrained[0:totalNeeded]))
      elif execType == 'aluncert':
            sInd = int(totalNeeded / 2)
            indices = indicesTobeTrained[0:sInd]
            nInd = totalNeeded - sInd
            indices.extend(indicesTobeTrained[len(indicesTobeTrained) - nInd:])
            return list(np.sort(indices))

   def getGMMIndicesfromEntropyPDFs(self,densityPDFs,densityProbs,learnInstances,instIndices):
          gMMClusters = {}
          probIndEntropy = []
          for k in learnInstances.keys():
           indVs = learnInstances[k]
           instInd = indVs[0]
           probInds = [float('%.5f' % prob) for prob in densityProbs[instInd]]
           probInds = [float(prob) for prob in probInds if prob > 0.0]
           entropyPdfs = [-1 * prob * np.log(prob) for prob in probInds]
           probIndEntropy.append(sum(entropyPdfs))
          argSortedEntropy = np.argsort(probIndEntropy)
          indicesTobeTrained = []
          learnInstancesNp = np.array(learnInstances.keys())
          if execType == 'alpool':
              orderedInstances = learnInstancesNp[argSortedEntropy]
              availNo =  sum([len(instIndices[inst]) for inst in orderedInstances])
              while availNo > len(indicesTobeTrained):
                  for inst in orderedInstances:
                    if len(instIndices[inst]) > 0:
                          ars = instIndices[inst]
                          popVal = ars.pop(0)
                          indicesTobeTrained.append(popVal)
                          instIndices[inst] = ars

          elif execType == 'aluncert':
             uncertCount = sum([1 for prob in probIndEntropy if prob > 0.0])
             certainIndices = argSortedEntropy[:len(probIndEntropy) - uncertCount]
             uncertainIndices = argSortedEntropy[len(probIndEntropy) - uncertCount:]
             certainInstances = learnInstancesNp[certainIndices]
             uncertainInstances = learnInstancesNp[uncertainIndices]   
             availNo =  sum([len(instIndices[inst]) for inst in certainInstances])
             while availNo > len(indicesTobeTrained):
                  for inst in certainInstances:
                    if len(instIndices[inst]) > 0:
                          ars = instIndices[inst]
                          popVal = ars.pop(0)
                          indicesTobeTrained.append(popVal)
                          instIndices[inst] = ars             
             uncIndices = []
             availNo =  sum([len(instIndices[inst]) for inst in uncertainInstances])
             while availNo > len(uncIndices):
                  for inst in uncertainInstances:
                    if len(instIndices[inst]) > 0:
                          ars = instIndices[inst]
                          popVal = ars.pop(0)
                          uncIndices.append(popVal)
                          instIndices[inst] = ars
             uncIndices = np.array(uncIndices)
             indicesTobeTrained.extend(uncIndices[::-1])
          return indicesTobeTrained

   def getGMMIndicesfromDenistyPDFs(self,densityPDFs,densityProbs,learnInstances,instIndices):
          gMMClusters = {}
          instOnes = 0
          for k in learnInstances.keys():
           indVs = learnInstances[k]
           instInd = indVs[0]
           myInd = np.argmax(densityPDFs[instInd])
           myVal = float(densityPDFs[instInd][myInd])
           if myInd in gMMClusters.keys():
               xx = gMMClusters[myInd]
               xx.update({k:myVal})
               gMMClusters[myInd] = xx
           else:
               gMMClusters[myInd] = {k:myVal}
          indicesTobeTrained = []
          if execType == 'alpool':
             availNo =  sum([len(instIndices[k]) for k in learnInstances.keys()])
             gMMClusters1 = {}
             for cNo in gMMClusters.keys():
                 cValues = gMMClusters[cNo]
#                 cValues = collections.OrderedDict(sorted(cValues.items(), reverse=True))
                 cValues = sorted(cValues.iteritems(), key=lambda (k,v): (v,k),reverse=True)
                 gMMClusters1[cNo] = [k[0] for k in cValues]
             gMMClusters = gMMClusters1
             gMMInstances = []
             while len(gMMInstances) < len(learnInstances.keys()):
              for cNo in gMMClusters.keys():
               cValues = gMMClusters[cNo]
               if len(cValues) > 0:
                 gMMInstances.append(cValues.pop(0))
                 gMMClusters[cNo] = cValues

             while availNo > len(indicesTobeTrained):
               for cVal in gMMInstances:
                    if len(instIndices[cVal]) > 0:
                      ars = instIndices[cVal]
                      indicesTobeTrained.append(ars.pop(0))
                      instIndices[cVal] = ars
             print "No of indices ",len(indicesTobeTrained)
          elif execType == 'aluncert':
                certainPoints = {}
                uncertainPoints = {}
                for cNo in gMMClusters.keys():
                 cValues = gMMClusters[cNo]
                 tPnts = {}
                 utPnts = {}
                 maxGMM = max(cValues.values())
                 minGMM = min(cValues.values())
                 avgGMM = (maxGMM + minGMM)/2
                 for kk,val in cValues.items():
                   if float(val) < float(avgGMM):
                        utPnts[kk] = val
                   else:
                        tPnts[kk] = val
                 tPnts = sorted(tPnts.iteritems(), key=lambda (k,v): (v,k),reverse=True)
                 utPnts = sorted(utPnts.iteritems(), key=lambda (k,v): (v,k),reverse=True)
                 certainPoints[cNo] = [key[0] for key in tPnts]
                 uncertainPoints[cNo] = [key[0] for key in  utPnts]
                indicesTobeTrained = []
                availNo =  sum([len(instIndices[inst]) for cNo in gMMClusters.keys() for inst in certainPoints[cNo]])
                countIndex = 0
                while availNo > len(indicesTobeTrained):
                  for cNo in gMMClusters.keys():
                        cVal = certainPoints[cNo]
                        inst = cVal[countIndex % len(cVal)]
                        if len(instIndices[inst]) > 0:
                          ars = instIndices[inst]
                          popVal = ars.pop(0)
                          indicesTobeTrained.append(popVal)
                          instIndices[inst] = ars
                  countIndex =  countIndex + 1
                availNo =  sum([len(instIndices[inst]) for cNo in gMMClusters.keys() for inst in uncertainPoints[cNo]])
                uncIndices = []
                countIndex = 0
                while availNo > len(uncIndices):
                  for cNo in gMMClusters.keys():
                    cVal = uncertainPoints[cNo]
                    if len(cVal) > 0:
                     inst = cVal[countIndex % len(cVal)]
                     if len(instIndices[inst]) > 0:
                          ars = instIndices[inst]
                          popVal = ars.pop(0)
                          uncIndices.append(popVal)
                          instIndices[inst] = ars
                  countIndex = countIndex + 1
                uncIndices = np.array(uncIndices)
                indicesTobeTrained.extend(uncIndices[::-1])
          return indicesTobeTrained

   def calculateGaussianProbability(self,data,mu,cov,weights):
      probs = []
      for dInst in data:
        prob = []
        for i in range(mu.shape[0]):
          N1 = 1/((2 * np.pi * np.linalg.det(cov[i])) ** (1/2))
          N2 = (-1/2) * (dInst - mu[i]).T.dot(np.linalg.inv(cov[i])).dot(dInst - mu[i])
          N = float(N1 * np.exp(N2))
          N = math.log(N1) + N2
          prob.append(N + math.log(weights[i]))
#        sumProb = math.log(np.sum([math.exp(p) for p in prob]))
        sumProb =  logsumexp(prob)
        prob = [pb - sumProb for pb in prob]
        probs.append(prob)
      return probs

   def generateALTrainingInstancesPool(self,nDf,tests,kind,totalNeeded):
      global noComponents
      df = read_table(self.annotationFile, sep=',',  header=None)
      cDz = df.values
      instances = nDf.to_dict()
      if totalNeeded >= len(cDz):
         return range(len(cDz))
      instIndices = {}
      instFiles = []
      for column in cDz:
          instFiles.append(column[0])

      for instName in instances.keys():
         inds = np.argwhere(np.array(instFiles)==instName)
         ars = [ind[0] for ind in inds]
         instIndices[instName] = ars

      data = []
      dataInsts = []
      for instName in instances.keys():
        if instName not in tests:
           instsData = instances[instName][0].getFeatures(kind)
           data.append(instsData[0])
           dataInsts.append(instName)
      learnInstances = {}
      for instName in list(set(dataInsts)):
          inds = np.argwhere(np.array(dataInsts)==instName)
          ars = [ind[0] for ind in inds]
          learnInstances[instName] = ars
      trainInstIndices = len(dataInsts)
      data = np.array(data)
      relPoints = trainInstIndices
      relDataSets = []
      if relPoints > 0:
#          gmm = GMM(noComps = gmmPoints,data = data)
#          gmm.em(data)
          gmm = mixture.GaussianMixture(n_components=gmmPoints, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='random', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)
          gmm.fit(data)
#          filename = 'gmm_model_rgb.pkl'
#          pickle.dump(gmm, open(filename, 'wb'))
          densityProbs = gmm.predict_proba(data)
#          densLogs = self.calculateGaussianProbability(data,gmm.means_,gmm.covariances_,gmm.weights_)
#          exit(0)
          centers = []
          densityPDFs = {}
#          nData = [dataFull[learnInstances[k][0]] for k in learnInstances.keys()]
          strdensPdfs = ""
#          exit(0)
          for i in range(gmm.n_components):
#            print gmm.weights_[i]
            density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(data)
            for idx,dens in enumerate(density):
              densityPDFs.setdefault(idx, []).append(density[idx])
          for idx in densityPDFs.keys():
             strIdx = [str(id) for id in densityPDFs[idx]]
             strdensPdfs += " ".join(strIdx)
             strdensPdfs += "\n"
          os.system("mkdir -p GMMIndices")
          f = open("GMMIndices/DensityPDFsGMMBased.log","w")
          f.write(strdensPdfs)
          f.close()
          for k in learnInstances.keys():
           indVs = learnInstances[k]
           instInd = indVs[0]
           myInd = np.argmax(densityPDFs[instInd])
           kk = k.split("/")
           xx = [str(b) for b in data[instInd]]
#           print kk[0],",",kk[1],",",",".join(xx),",\'",str(myInd + 1) ,"\'",densLogs[instInd]

          strdensPdfs = ""
          for idx in densityProbs:
             strIdx = [str(id) for id in idx]
             strdensPdfs += " ".join(strIdx)
             strdensPdfs += "\n"
          f = open("GMMIndices/predictProbGMMBased.log","w")
          f.write(strdensPdfs)
          f.close()
#          print "<table>"
#          print "<tr><td>Number of Components :: </td><td>" + str(gmm.n_components) + "</td></tr>"
#          print "<tr><td>ObjectName, </td><td>Instance Name,</td><td> GMM Cluster Probability Densities,</td><td>Cluster based on Densities, </td><td>GMM Cluster Probabilities,</td><td> Cluster Number based on probabilities </td></tr>"
          clusterImgs = {}
          for idx in densityPDFs.keys():
              instS = dataInsts[idx].split("/")
#              print "<tr><td>",instS[0],",</td><td>",dataInsts[idx],",</td><td>",
              strIdx = [str(id) for id in densityPDFs[idx]]
#              print " ".join(strIdx),",</td><td>",
#              print str(np.argmax(densityPDFs[idx])),",</td><td>",
              strIdx = [str(id) for id in densityProbs[idx]]
#              print " ".join(strIdx),",</td><td>",
#              print str(np.argmax(densityProbs[idx])),"</td></tr>"
              clusterImgs.setdefault(np.argmax(densityPDFs[idx]), []).append(dataInsts[idx])
#          print "</table>"
#          print "<table><tr><th> Clusters based on Densities</th></tr>"

          for key, imgs in clusterImgs.items():
#            print "<tr><td>",str(key),"<td>"
            for img in imgs:
               oN = img.split("/")
#               print '<td><img height="42" width="42" src="https://raw.githubusercontent.com/nispillai/DataSet-GroundedLanguageAcquisition/master/UMBC-RGB-DATASET/' + str(img) + '/' + str(oN[1]) + '_1.png"></td>'
#            print "</tr>"
#          print "/table>"

          if selectionType == 'max':
            indicesTobeTrained = self.getGMMIndicesfromDenistyPDFs(densityPDFs,densityProbs,learnInstances,instIndices)
          elif selectionType == 'entropy':
            indicesTobeTrained = self.getGMMIndicesfromEntropyPDFs(densityPDFs,densityProbs,learnInstances,instIndices)
          strIndicesToBeTrained = [str(ind) for ind in indicesTobeTrained]
          lineIndices = " ".join(strIndicesToBeTrained)
          os.system("echo \'" + lineIndices + "\' >  GMMIndices/indicesGMMbased.log")
#          print len(indicesTobeTrained),len(list(set(indicesTobeTrained)))
          if totalNeeded > len(indicesTobeTrained):
             return list(np.sort(indicesTobeTrained))
          if execType == 'alpool':
            return list(np.sort(indicesTobeTrained[0:totalNeeded]))
          elif execType == 'aluncert':
            sInd = int(totalNeeded / 2)
            indices = indicesTobeTrained[0:sInd]
            nInd = totalNeeded - sInd
            indices.extend(indicesTobeTrained[len(indicesTobeTrained) - nInd:])
            return list(np.sort(indices))



   def generateALTrainingInstancesPool1(self,nDf,tests,kind,totalNeeded):
      global noComponents
#      print "Acquiring AL testcase indices"
      df = read_table(self.annotationFile, sep=',',  header=None)
      cDz = df.values
      instances = nDf.to_dict()
      if totalNeeded >= len(cDz):
         return range(len(cDz))
      instIndices = {}
      instFiles = []
#      print cDz
      for column in cDz:
          instFiles.append(column[0])

      for instName in instances.keys():
         inds = np.argwhere(np.array(instFiles)==instName)
         ars = [ind[0] for ind in inds]
         instIndices[instName] = ars

      data = []
      dataInsts = []
#      for idx,column in enumerate(cDz):
#        instName = column[0]
      for instName in instances.keys():
        if instName not in tests:
           instsData = instances[instName][0].getFeatures(kind)
           data.append(instsData[0])
           dataInsts.append(instName)
#      dataFull = [list(instances[column[0]][0].getFeatures(kind)[0]) for idx,column in enumerate(cDz) if column[0] not in tests]
#      dataFull = [list(inst) for instName in instances.keys() if instName not in tests for inst in instances[instName][0].getFeatures(kind)]
#      dataInsts = [column[0] for idx,column in enumerate(cDz) if column[0] not in tests]
#      dataInsts = [instName for instName in instances.keys() if instName not in tests for inst in instances[instName][0].getFeatures(kind)]
#      trainInstIndices = sum([len( instIndices[instName]) for instName in instances.keys() if instName not in tests])
      learnInstances = {}
      for instName in list(set(dataInsts)):
          inds = np.argwhere(np.array(dataInsts)==instName)
          ars = [ind[0] for ind in inds]
          learnInstances[instName] = ars
      trainInstIndices = len(dataInsts)
      data = np.array(data)
      data = preprocessing.normalize(data, norm='l2',axis=1)
      relPoints = trainInstIndices
      relDataSets = []
      if relPoints > 0:
          gmm = mixture.GaussianMixture(n_components=gmmPoints, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='random', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)
          gmm.fit(data)
#          filename = 'gmm_model_rgb.pkl'
#          pickle.dump(gmm, open(filename, 'wb'))
          densityProbs = gmm.predict_proba(data)
#          densLogs = self.calculateGaussianProbability(data,gmm.means_,gmm.covariances_,gmm.weights_)
#          exit(0)
          centers = []
          densityPDFs = {}
#          nData = [dataFull[learnInstances[k][0]] for k in learnInstances.keys()]
          strdensPdfs = ""
#          exit(0)
          for i in range(gmm.n_components):
#            print gmm.weights_[i]
            density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(data)
            for idx,dens in enumerate(density):
              densityPDFs.setdefault(idx, []).append(density[idx])
          for idx in densityPDFs.keys():
             strIdx = [str(id) for id in densityPDFs[idx]]
             strdensPdfs += " ".join(strIdx)
             strdensPdfs += "\n"
          f = open("GMMIndices/DensityPDFsGMMBased.log","w")
          f.write(strdensPdfs)
          f.close()
          for k in learnInstances.keys():
           indVs = learnInstances[k]
           instInd = indVs[0]
           myInd = np.argmax(densityPDFs[instInd])
           kk = k.split("/")
           xx = [str(b) for b in data[instInd]]
#           print kk[0],",",kk[1],",",",".join(xx),",\'",str(myInd + 1) ,"\'",densLogs[instInd]

          strdensPdfs = ""
          for idx in densityProbs:
             strIdx = [str(id) for id in idx]
             strdensPdfs += " ".join(strIdx)
             strdensPdfs += "\n"
          f = open("GMMIndices/predictProbGMMBased.log","w")
          f.write(strdensPdfs)
          f.close()
          print "<table>"  
          print "<tr><td>Number of Components :: </td><td>" + str(gmm.n_components) + "</td></tr>"
          print "<tr><td>ObjectName, </td><td>Instance Name,</td><td> GMM Cluster Probability Densities,</td><td>Cluster based on Densities, </td><td>GMM Cluster Probabilities,</td><td> Cluster Number based on probabilities </td></tr>"
          clusterImgs = {}
          for idx in densityPDFs.keys():
              instS = dataInsts[idx].split("/")
              print "<tr><td>",instS[0],",</td><td>",dataInsts[idx],",</td><td>",
              strIdx = [str(id) for id in densityPDFs[idx]]
              print " ".join(strIdx),",</td><td>",
              print str(np.argmax(densityPDFs[idx])),",</td><td>",
              strIdx = [str(id) for id in densityProbs[idx]]
              print " ".join(strIdx),",</td><td>",
              print str(np.argmax(densityProbs[idx])),"</td></tr>"
              clusterImgs.setdefault(np.argmax(densityPDFs[idx]), []).append(dataInsts[idx])
          print "</table>"
          print "<table><tr><th> Clusters based on Densities</th></tr>"
          
          for key, imgs in clusterImgs.items():
            print "<tr><td>",str(key),"<td>"
            for img in imgs:
               oN = img.split("/")
               print '<td><img height="42" width="42" src="https://raw.githubusercontent.com/nispillai/DataSet-GroundedLanguageAcquisition/master/UMBC-RGB-DATASET/' + str(img) + '/' + str(oN[1]) + '_1.png"></td>'
            print "</tr>"          
          print "/table>"
           
          exit(0)
          if selectionType == 'max':
            indicesTobeTrained = self.getGMMIndicesfromDenistyPDFs(densityPDFs,densityProbs,learnInstances,instIndices)
          elif selectionType == 'entropy':
            indicesTobeTrained = self.getGMMIndicesfromEntropyPDFs(densityPDFs,densityProbs,learnInstances,instIndices)
          strIndicesToBeTrained = [str(ind) for ind in indicesTobeTrained]
          lineIndices = " ".join(strIndicesToBeTrained)
          os.system("echo \'" + lineIndices + "\' >  GMMIndices/indicesGMMbased.log")
          print len(indicesTobeTrained),len(list(set(indicesTobeTrained)))
          if totalNeeded > len(indicesTobeTrained):
             return list(np.sort(indicesTobeTrained))
          if execType == 'alpool':
            return list(np.sort(indicesTobeTrained[0:totalNeeded]))
          elif execType == 'aluncert':
            sInd = int(totalNeeded / 2)
            indices = indicesTobeTrained[0:sInd]
            nInd = totalNeeded - sInd
            indices.extend(indicesTobeTrained[len(indicesTobeTrained) - nInd:])
            return list(np.sort(indices))

   def getALTrainingInstancesOld(self,nDf,tests,kind,totalNeeded):
      global noComponents
#      print "Acquiring AL testcase indices"
      df = read_table(self.annotationFile, sep=',',  header=None)
      cDz = df.values
      instances = nDf.to_dict()
      if totalNeeded >= len(cDz):
         return range(len(cDz))
      instIndices = {}
      instFiles = []
#      print cDz
      for column in cDz:
          instFiles.append(column[0])

      for instName in instances.keys():

         inds = np.argwhere(np.array(instFiles)==instName)
         ars = [ind[0] for ind in inds]
         instIndices[instName] = ars

      data = [list(inst) for instName in instances.keys() if instName not in tests for inst in instances[instName][0].getFeatures(kind)]
      
      dataInsts = [instName for instName in instances.keys() if instName not in tests for inst in instances[instName][0].getFeatures(kind)]
      data = np.array(data)
      print data.shape
      relPoints = totalNeeded
      relDataSets = []
      print relPoints,len(dataInsts)
      if relPoints >= len(dataInsts):
        relDataSets = [instFiles[i % len(instFiles)] for i in range(relPoints)]
      else:
          gmm = mixture.GaussianMixture(n_components=15, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)
          gmm.fit(data)
          centers = []
          uncertainPts = []
          densityOrder = []
          for i in range(gmm.n_components):
            density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(data)
            dens1 = np.argsort(density)
            densityOrder.append(dens1)
            if totalNeeded >= 30:
                 lnUncts = len(density) - 20
                 unDensPts = dens1[0:lnUncts]
                 if i == 0:
                     uncertainPts = unDensPts
                 else:
                     uncertainPts = list(set(uncertainPts).intersection(set(unDensPts)))
          if totalNeeded >= 30:
             upLength = totalNeeded / 2
             rDS = list(set([dataInsts[i] for i in uncertainPts]))
             if len(rDS) < upLength:
                  upLength = len(rDS)
             
             relDataSets.extend([i for i in rDS[0:upLength]])      
             
             totalNeeded = totalNeeded - upLength
          for i in range(gmm.n_components):
              cmptNeeded = totalNeeded / gmm.n_components
              if i < (totalNeeded % gmm.n_components):
                cmptNeeded += 1
              dens1 = densityOrder[i]
              dens1 = np.flipud(dens1)
              den1 = dens1[0:cmptNeeded]
              centers.append(den1)
              relDataSets.extend([dataInsts[i] for i in den1])
      indicesTobeTrained = []
      for inst in relDataSets:
        ars = instIndices[inst]
        indicesTobeTrained.append(ars.pop(0))
        instIndices[inst] = ars
      indicesTobeTrained = list(np.sort(list(set(indicesTobeTrained))))

      return indicesTobeTrained


   def getALTrainingInstances1(self,nDf,tests,kind,totalNeeded):
      
      epsilon = 0
      if kind == 'rgb':
        epsilon = 35
      elif kind == 'shape':
        epsilon = 1200
      else :
        epsilon = 1000
      df = read_table(self.annotationFile, sep=',',  header=None)
      cDz = df.values
      instances = nDf.to_dict()
      if totalNeeded >= len(cDz):
         return range(len(cDz))      
      data = [list(inst) for column in df.values for inst in instances[column[0]][0].getFeatures(kind)]
      data = np.array(data)
      indices = [i for (i,column) in enumerate(df.values) for ii in range(len(instances[column[0]][0].getFeatures(kind)))]
      trainInds = [i for (i,column) in enumerate(df.values) if column[0] not in tests] 
      if totalNeeded >= len(trainInds):
         return trainInds      
#      db = DBSCAN(eps=epsilon, min_samples=totalNeeded).fit(data)
      cls = 10 
      if len(set(indices)) <= cls:
        cls = 3
      db = AgglomerativeClustering(n_clusters=cls).fit(data)
      labels = db.labels_
      n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
      if n_clusters_ == 0:
           indicesTobeTrained = []
           ind = 0
           indices1 = list(set(trainInds))
           random.shuffle(indices1)
           while(len(indicesTobeTrained) < totalNeeded):
              # ind = random.randint(0,len(indices1) - 1)
               ind = ind % len(indices1)
               indicesTobeTrained.append(indices1[ind])
               ind += 1
           return np.sort(indicesTobeTrained)     
      indicesToPick = totalNeeded/n_clusters_
      indicesTobeTrained = []
      for ind in set(labels):
         if ind != -1:
            indexes = [x for x in range(len(labels)) if labels[x]==ind]
            setIndices = list(set([indices[k] for k in indexes]))
            setIndices = [x  for x in setIndices if x in trainInds]
            random.shuffle(setIndices)
            indPick = indicesToPick
            if len(setIndices) < indPick:
                 indPick = len(setIndices)
            indds = [setIndices[i] for i in range(indPick)]
            indicesTobeTrained.extend(indds) 
      
      trainInds = [ x for x in trainInds if x not in indicesTobeTrained]
      random.shuffle(trainInds)
      while(len(indicesTobeTrained) < totalNeeded): 
         indicesTobeTrained.append(trainInds[0]) 
      return np.sort(indicesTobeTrained)


   def getDataSet(self,cDf,nDf,tests,tIndices,fName):
      """""""""""""""""""""""""""""""""""""""""
        Function to add amazon mechanical turk description file, 
        find all tokens, find positive and negative instances for all tokens

             Args:  dataset instance, array of Category class instances, 
             	array of Instance class instances, array of instance names to test, 
             	file name for logging
             Returns:  array of Token class instances
      """""""""""""""""""""""""""""""""""""""""     	   
      instances = nDf.to_dict()
      """ read the amazon mechanical turk description file line by line, 
      separating by comma [ line example, 'arch/arch_1, yellow arch' """
      df = read_table(self.annotationFile, sep=',',  header=None)  
      tokenDf = {}
      cDz = df.values
      """ column[0] would be arch/arch_1 and column[1] would be 'yellow arch' """ 
      docs = {}
#      for column in df.values:
      for ind in tIndices:
       if ind < len(cDz): 
        column = cDz[ind]
        ds = column[0]
        if ds in docs.keys():
           sent = docs[ds]
           sent += " " + column[1]
           docs[ds] = sent
        else:
           docs[ds] = column[1]
        dsTokens = column[1].split(" ")
        dsTokens = list(filter(None, dsTokens)) 
        """ add 'yellow' and 'arch' as the tokens of 'arch/arch_1' """
        instances[ds][0].addTokens(dsTokens)
        if ds not in tests:
         iName = instances[ds][0].getName()
         for annotation in dsTokens:
             if annotation not in tokenDf.keys():
                 """ creating Token class instances for all tokens (ex, 'yellow' and 'arch') """
                 tokenDf[annotation] = Token(annotation)
             """ add 'arch/arch_1' as a positive instance for token 'yellow' """    
             tokenDf[annotation].extendPositives(ds) 
      tks = pd.DataFrame(tokenDf,index=[0])
      sent = "Tokens :: "+ " ".join(tokenDf.keys())
      fileAppend(fName,sent)
      negSelection = NegSampleSelection(docs)
      negExamples = negSelection.generateNegatives()
      """ find negative instances for all tokens.
      Instances which has cosine angle greater than 80 in vector space consider as negative sample"""
      for tk in tokenDf.keys():
         poss = list(set(tokenDf[tk].getPositives()))
         negs = []
         for ds in poss:
             if isinstance(negExamples[ds], str):
                  negatives1 = negExamples[ds].split(",")
		  negatives = []
                  for instNeg in negatives1:
                       s1 = instNeg.split("-")
                       """ if the cosine angle between 2 instances is greater that dgAbove (80 in this case),
                       we consider them as negative instances to each other """
                       if int(float(s1[1])) >= dgAbove:
                             negatives.append(s1[0])
                  negatives = [xx for xx in negatives if xx not in tests]
#                  negatives = [xx for xx in negatives if xx not in poss]
                  negs.extend(negatives)
         negsPart = []
         for part in quartiles:
            noElements  = len(negs)/ sections
            sNo = (part - 1)* noElements          
#	    eNo = part  * noElements
#	    if part == sections:
	    eNo =  len(negs)
	    kk = negs[sNo:eNo]
         negsPart.extend(kk)
         negsPart = negs
         tokenDf[tk].extendNegatives(negsPart)
      return (nDf,tks,tests) 

              
def getTestFiles(insts,kind,tests,token):
   """""""""""""""""""""""""""""""""""""""""
   Function to get all feature sets for testing and dummy 'Y' values
   		Args:  Array of all Instance class instances, type of testing  
   				(rgb, shape, or object) , array of test instance names, 
   				token (word) that is testing
        Returns:  Feature set and values for testing
   """""""""""""""""""""""""""""""""""""""""    	
   instances = insts.to_dict()
   features = []
   y = []
   for nInst in tests:
      y1 = instances[nInst][0].getY(token,kind)
      fs  = instances[nInst][0].getFeatures(kind)
      features.append(list(fs))
      y.append(list(np.full(len(fs),y1)))
   return(features,y)


def findTrainTestFeatures(insts,tkns,tests):
  """""""""""""""""""""""""""""""""""""""""
  Function to iterate over all tokens, find train and test features for execution
  	Args:  Array of all Instance class instances, 
  		array of all Token class instances, 
  		array of test instance names
    Returns:  all train test features, values, type of testing
  """""""""""""""""""""""""""""""""""""""""    	
  tokenDict = tkns.to_dict()
  for token in np.sort(tokenDict.keys()):
#  for token in ['arch']:
     objTkn = tokenDict[token][0]
     for kind in kinds:
#     for kind in ['rgb']: 
        (features,y) = objTkn.getTrainFiles(insts,kind)
        (testFeatures,testY) = getTestFiles(insts,kind,tests,token)
        if len(features) == 0 :
            continue;
        yield (token,kind,features,y,testFeatures,testY)


def callML(resultDir,insts,tkns,tests,algType,resfname):
  """ generate a CSV result file with all probabilities 
	for the association between tokens (words) and test instances"""	
  confFile = open(resultDir + '/groundTruthPrediction.csv','w')
  headFlag = 0
  fldNames = np.array(['Token','Type'])
  confWriter = csv.DictWriter(confFile, fieldnames=fldNames)
  
  """ Trying to add correct object name of test instances in groundTruthPrediction csv file
  	ex, 'tomato/tomato_1 - red tomato' """
  featureSet = read_table(fAnnotation,sep=',',  header=None)
  featureSet = featureSet.values
  fSet = dict(zip(featureSet[:,0],featureSet[:,1]))
  testTokens = []
  
  """ fine tokens, type to test, train/test features and values """
  for (token,kind,X,Y,tX,tY) in findTrainTestFeatures(insts,tkns,tests):
   if token not in testTokens:
      testTokens.append(token)
   print "Token : " + token + ", Kind : " + kind
   """ binary classifier Logisitc regression is used here """
   polynomial_features = PolynomialFeatures(degree=2,include_bias=False)
   sgdK = linear_model.LogisticRegression(C=10**5,random_state=0)
#   pipeline2_2 = Pipeline([("polynomial_features", polynomial_features),
#                         ("logistic", sgdK)])
   pipeline2_2 = Pipeline([("logistic", sgdK)])


   if algType == 0:
     pipeline2_2.fit(X,Y)

   fldNames = np.array(['Token','Type'])  
   confD = {}
   ttX = []
   ttY = []
   testTT = []
   confDict = {'Token' : token,'Type' : kind}
   """ testing all images category wise and saving the probabilitties in a Map 
   		for ex, for category, tomato, test all images (tomato image 1, tomato image 2...)"""
   for ii in range(len(tX)) :
      testX = tX[ii]
      testY = tY[ii]
      ttX.extend(testX)
      ttY.extend(testY)
      tt = tests[ii]
      for ik in range(len(testY)):
         fldNames = np.append(fldNames,str(ik) + "-" + tt)
         confD[str(ik) + "-" + tt] = str(fSet[tt])
         testTT.append(str(ik) + "-" + tt)

   predY = []  
   tProbs = []
   if algType == 0:
         predY = pipeline2_2.predict(ttX)
         acc = pipeline2_2.score(ttX, ttY)
         probK = pipeline2_2.predict_proba(ttX)
         tProbs = probK[:,1]
   for ik in range(len(tProbs)):
           confDict[testTT[ik]] = str(tProbs[ik])


   if headFlag == 0:
      headFlag = 1
      """ saving the header of CSV file """
      confWriter = csv.DictWriter(confFile, fieldnames=fldNames)
      confWriter.writeheader()
      confWriter.writerow(confD)
   """ saving probabilities in CSV file """
   confWriter.writerow(confDict)

  confFile.close()

def setALParameters():
  global gmmPoints,gmmGenerate,iteractionNo,selectionType,dppPts
  gmmPoints = args.gmmPts 
  gmmGenerate = args.generate
  iteractionNo = args.iteration
  selectionType = args.selectionType
  dppPts = args.dppPts

#def execution(resultDir,ds,cDf,nDf,tests):
def execution(inds,resultDir,ds,cDf,nDf,tests,kind,totalTrainCases):
	
    resultDir1 = resultDir + "/NoOfDataPoints/" + str(inds)
    os.system("mkdir -p " + resultDir1)
    tIndices = totalTrainCases
    if execType == 'random' or execType == 'seq':
      if execType == 'random':
        random.seed(4)
        random.shuffle(tIndices)
      if inds < len(totalTrainCases):
        tIndices = tIndices[0:inds]
    elif execType == 'alpool' or execType == 'aluncert':
       setALParameters()
       if gmmGenerate:
         tIndices = ds.generateALTrainingInstancesPool(nDf,tests,kind,inds)
       else:
         tIndices = ds.getALTrainingInstancesPool(nDf,tests,kind,inds)
    elif execType == 'aldpp' or execType == 'algmmdpp':
       setALParameters()
       if gmmGenerate:
         tIndices = ds.generateALDPPTrainingInstances(nDf,tests,kind,inds)
       else:
         tIndices = ds.getALDPPTrainingInstances(nDf,tests,kind,inds)
    fResName = resultDir1 + "/results.txt"
    sent = "Test Instances :: " + " ".join(tests)
#    fileAppend(fResName,sent)
    """ read amazon mechanical turk file, find all tokens
    get positive and negative instance for all tokens """
    (insts,tokens,tests) = ds.getDataSet(cDf,nDf,tests,tIndices,fResName)
#    tokens = ds.getDataSet(cDf,nDf,tests,fResName)

    """ Train and run binary classifiers for all tokens, find the probabilities 
    	for the associations between all tokens and test instances, 
    	and log the probabilitties """
#    callML(resultDir1,nDf,tokens,tests,0,fResName)
    callML(resultDir1,insts,tokens,tests,0,fResName)

def getTotalTrainCases(anFile,tests,nDf,kind):
   totalTrainCases = 0
   df = read_table(anFile, sep=',',  header=None)
   cDz = df.values
   instances = nDf.to_dict()
#   totalTrainCases = [idx for idx,column in enumerate(cDz) if column[0] not in tests for inst in instances[column[0]][0].getFeatures(kind)]
   totalTrainCases = [idx for idx,column in enumerate(cDz) if column[0] not in tests]
   return totalTrainCases

if __name__== "__main__":
  print "START :: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  anFile =  execPath + "groundtruth_annotation.conf"
  anFile =  execPath + preFile
  fResName = ""
  os.system("mkdir -p " + resultDir)
  """ creating a Dataset class Instance with dataset path, amazon mechanical turk description file"""
  ds = DataSet(dsPath,anFile)
  """ find all categories and instances in the dataset """
  (cDf,nDf) = ds.findCategoryInstances()
  """ find all test instances. We are doing 4- fold cross validation """
  tests = ds.splitTestInstances(cDf)
  print "Script START :: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  kind = kinds[0]
  totalTrainCases = getTotalTrainCases(anFile,tests,nDf,kind)
  execution(numberOfDPs,resultDir,ds,cDf,nDf,tests,kind,totalTrainCases)
  
#  execution(resultDir,ds,cDf,nDf,tests)
  print "Script END :: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
