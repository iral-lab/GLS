#!/usr/bin/env python
import numpy as np
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
from pandas import DataFrame, read_table
import pandas as pd
import collections
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
from sklearn.preprocessing import MultiLabelBinarizer
import sklearn
import argparse
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from scipy import spatial
from ActiveLearning import AL
from RepresentationLearning import VAE
from CNN.cLLMultiLabel import MultiLabel
mLabel = MultiLabel()

import yamale

parser = argparse.ArgumentParser()
parser.add_argument('--confFile', help='configuration filename',required=True)
parser.add_argument('--schema', help='filename of schema',required=True)
parser.add_argument('--resDir',help='path to result directory',required=True)
args = parser.parse_args()


schema = yamale.make_schema(args.schema)
confFile = yamale.make_data(args.confFile)
# Validate data against the schema. Throws a ValueError if data is invalid.
yamale.validate(schema, confFile, strict=True )
configParams = confFile[0][0]


resultDir = args.resDir
preFile = configParams['annotationFile']
kind = configParams['cat']
negSampling = configParams['negatives']
execType = configParams['execType']
numberOfDPs = configParams['noPts']
dsPath = configParams['featuresDSPath']
fAnnotation = configParams['fAnnot']
classModel = configParams['classifierModel']
ablationPercentage = configParams['ablationPercentage']
inputFeatures = configParams['inputFeatures']
nnName = configParams['cnnChoice']
cNNTune = configParams['cnnFineTuning']
imagePath = configParams['imageDSPath']
allImgForVAE = configParams['VAE']['allImagesForVAE']
cumulativeFeatures = configParams['VAE']['cumulativeFeatures']
dgAbove = 80

ds = ""
cDf = ""
nDf = "" 
tests = ""

cNNInputFeatures = {}
allInstTokens = {}
allInstDicts = pd.DataFrame()
generalColors = ['yellow','blue','purple','black','isyellow','green','brown','orange','white','red']

generalObjs = ['potatoe','cylinder','square', 'cuboid', 'sphere', 'halfcircle','circle','rectangle','cube','triangle','arch','semicircle','halfcylinder','wedge','block','apple','carrot','tomato','lemon','cherry','lime', 'banana','corn','hemisphere','cucumber','cabbage','ear','potato', 'plantain','eggplant']

generalShapes = ['spherical', 'cylinder', 'square', 'rounded', 'cylindershaped', 'cuboid', 'rectangleshape','arcshape', 'sphere', 'archshaped', 'cubeshaped', 'curved' ,'rectangular', 'triangleshaped', 'halfcircle', 'globular','halfcylindrical', 'circle', 'rectangle', 'circular', 'cube', 'triangle', 'cubic', 'triangular', 'cylindrical','arch','semicircle', 'squareshape', 'arched','curve', 'halfcylinder', 'wedge', 'cylindershape', 'round', 'block', 'cuboidshaped']


rgbWords  = ['yellow','blue','purple', 'orange','red','green']
shapeWords  = ['cylinder','cube', 'triangle','triangular','rectangular']
objWords = ['cylinder', 'apple','carrot', 'lime','lemon','orange', 'banana','cube', 'triangle', 'corn','cucumber', 'half', 'cabbage', 'ear', 'tomato', 'potato', 'cob','eggplant']

tobeTestedTokens = rgbWords
tobeTestedTokens.extend(shapeWords)
tobeTestedTokens.extend(objWords)
global modelNN, mlbNN



def fileAppend(fName, sentence):
  """""""""""""""""""""""""""""""""""""""""
	Function to write results/outputs to a log file
	 	Args: file descriptor, sentence to write
	 	Returns: Nothing
  """""""""""""""""""""""""""""""""""""""""
  with open(fName, "a") as myfile:
    myfile.write(sentence)
    myfile.write("\n")

#############Classifiers#######################
def classifierModel(model="LR", xDim=700, yDim=2):
   pipeline = ()
   if model == "LR":
      polynomial_features = PolynomialFeatures(degree=2,include_bias=False)
      sgdK = linear_model.LogisticRegression(C=10**5,random_state=0, solver='liblinear')
      #pipeline2_2 = Pipeline([("polynomial_features", polynomial_features),
      #("logistic", sgdK)])
      pipeline = Pipeline([("logistic", sgdK)])
   elif model == "SVM":
      pipeline = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', probability=True)
   elif model == "MLP":    
      hDims = configParams['hiddenDims']
      activationFunction = configParams['activationFn']
#      pipeline, pExtract = MLP(xDim, yDim, hDims, activationFunction )
      pipeline = MLPClassifier(hidden_layer_sizes=(hDims), activation=activationFunction, alpha=1e-4, solver='adam', verbose=10, tol=1e-4, random_state=1, learning_rate_init=.1)
   return pipeline   	


def MLP(xDim, yDim, hDims, activationFunction):
   l2_loss = 1e-6
   input_shape = (xDim, )
   inputs = Input(shape=input_shape)   
   layer = inputs
   for i in range(len(hDims)):
   	   x = Dense(hDims[i], activation=activationFunction, kernel_regularizer = l2(l2_loss))(layer)
   	   layer = x       	 
   outs = Dense(yDim, activation='softmax')(layer)
# instantiate model
   model = Model(inputs=inputs, outputs=outs)
   modelExtract = Model(inputs=inputs, outputs=layer)
   model.summary()
   model.compile(optimizer='adam',loss='mean_squared_error')
   
   return model, modelExtract
   
def classifierFit(classModel, X, Y, model="lr"):

      if model == "LR" or model == "SVM":
              classModel.fit(X,Y)
      elif model == "MLP":    
              classModel.fit(X,Y)      
#              classModel.fit(X, Y, epochs=25,batch_size=1)

 	
         
def classifierPredict(classModel, testX, model="LR"):
      testX = np.array(testX)

      if model == "LR" or model == "SVM":
              return classModel.predict_proba(testX)
      elif model == "MLP":    
#               outs = classModel.predict(testX)      
#               return np.array([[1.0- i[0], i[0]] for i in outs])
              return classModel.predict_proba(testX)



######## Negative Example Generation##########
class LabeledLineSentence(object):
    def __init__(self,docLists,docLabels):
        self.docLists = docLists
        self.docLabels = docLabels

    def __iter__(self):
        for index, arDoc in enumerate(self.docLists):
#            yield LabeledSentence(arDoc, [self.docLabels[index]])
            yield TaggedDocument(words=arDoc, tags=[self.docLabels[index]])

    def to_array(self):
        self.sentences = []
        for index, arDoc in enumerate(self.docLists):
#            self.sentences.append(LabeledSentence(arDoc, [self.docLabels[index]]))
            self.sentences.append(TaggedDocument(words=arDoc, tags=[self.docLabels[index]]))
        return self.sentences
    
    def sentences_perm(self):
        from random import shuffle
        shuffle(self.sentences)
        return self.sentences

class NegSampleSelection:
   """ Class to bundle negative example generation functions and variables. """
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

   def getDoc2Vec(self, vectorSize=2000):
      docs = self.docs
      docNames = list(docs.keys())
      docLists = self.sentenceToWordLists()
      docDicts = self.sentenceToWordDicts()
      docLabels = []
      for key in docNames:
        ar = key.split("/")
        docLabels.append(ar[1])
      sentences = LabeledLineSentence(docLists,docLabels)
      model = Doc2Vec(min_count=1, window=10, vector_size=vectorSize, sample=1e-4, negative=5, workers=1)
      model.build_vocab(sentences.to_array())
#      token_count = sum([len(sentence) for sentence in sentences])
      token_count = len([sent for sent in sentences])
      for epoch in range(10):
          model.train(sentences.sentences_perm(),total_examples = token_count,epochs=100)
          model.alpha -= 0.002 
          model.min_alpha = model.alpha 
          model.train(sentences.sentences_perm(),total_examples = token_count,epochs=100)

      d2VecData = {}
      
      for i , item1 in enumerate(docLabels):
         fDoc = model.docvecs[docLabels[i]]
         d2VecData[docNames[i]] = fDoc
      return d2VecData
	  
	  
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
      model = Doc2Vec(min_count=1, window=10, vector_size=2000, sample=1e-4, negative=5, workers=1)

      model.build_vocab(sentences.to_array())
#      token_count = sum([len(sentence) for sentence in sentences])
      token_count = len([sent for sent in sentences])
      for epoch in range(10):
          model.train(sentences.sentences_perm(),total_examples = token_count,epochs=100)
          model.alpha -= 0.002 
          model.min_alpha = model.alpha 
          model.train(sentences.sentences_perm(),total_examples = token_count,epochs=100)

      degreeMap = {}
      docNames = list(docNames)
      docVecFeatures = {}
      for i , item1 in enumerate(docLabels):
         fDoc = model.docvecs[docLabels[i]]
         docVecFeatures[i] = fDoc
                     
      for i , item1 in enumerate(docLabels):
#         fDoc = model.docvecs[docLabels[i]]
         fDoc = docVecFeatures[i]
         cInstMap = {}
         cInstance = docNames[i]
         for j,item2 in enumerate(docLabels):
#            tDoc = model.docvecs[docLabels[j]]
            tDoc = docVecFeatures[j]
            cosVal = self.cosine_similarity(fDoc,tDoc)
            cosineVal = np.sign(cosVal) * min(abs(cosVal), float(1))
            cValue = math.degrees(math.acos(cosineVal))
            tInstance = docNames[j]
            cInstMap[tInstance] = cValue
         degreeMap[cInstance] = cInstMap
      negInstances = {}
      for k in sorted(list(degreeMap.keys())):
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

class Category:
   """ Class to bundle our dataset functions and variables category wise. """
   catNums = np.array([], dtype='object')
  
   def __init__(self, name):
      """""""""""""""""""""""""""""""""""""""""
		Initialization function for category class
     		Args: category name
     		Returns: Nothing
      """""""""""""""""""""""""""""""""""""""""     	
      self.name = name

   def getCatNums(self):
      """""""""""""""""""""""""""""""""""""""""
        Function to get the category Nums
                Args: category class instance
                Returns: category name
      """""""""""""""""""""""""""""""""""""""""
      return self.catNums

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

   def clearAndUpdateCategoryInstances(self,nums):
      """""""""""""""""""""""""""""""""""""""""
      Function to clear category numbers and update new list
         Args: category class instance
         Returns: None
      """""""""""""""""""""""""""""""""""""""""
      self.catNums = np.unique(nums)

   def chooseOneInstance(self):
      """""""""""""""""""""""""""""""""""""""""
      Function to select one random instance from this category for testing
         Args: category class instance
         Returns: Randomly selected instance name
      """""""""""""""""""""""""""""""""""""""""
      r = random.sample(range(0,self.catNums.size - 1),k=int(self.catNums.size/4)) 
#      r = random.randint(0,self.catNums.size - 1)
#      r= int(tCatNum)
      instNames = []
#      instName = self.name + "/" + self.name + "_" + self.catNums[r]
      for i in r:
          instNames.append(self.name + "/" + self.name + "_" + self.catNums[i])
     
      return instNames


class Instance(Category):
	""" Class to bundle instance wise functions and variables """
	gT = {}
	tokens = np.array([])
	name = ''
	negs = np.array([])
	cNNFeatures = np.array([])
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
		
	def getImages(self):
		"""""""""""""""""""""""""""""""""""""""""
		Function to find the complete dataset file path (.../arch/arch_1/arch_1_rgb.log)
		where the visual feaures are stored, read the features from the file, and return

         	Args: Instance class instance, type of features(rgb, shape, or object)
         	Returns: feature set
        """""""""""""""""""""""""""""""""""""""""
		instName = self.name
		instName.strip()
		ar1 = instName.split("/")
		images = []
		path1 = "/".join([imagePath,instName])

		for o1 in os.listdir(path1):
			if o1.endswith(".png"):
				od1 = os.path.join(path1, o1)
				images.append(od1)
		return images			        

	def getKernelDescriptorFeatures(self, kind):
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

	def getCNNFeatures(self):
		instName = self.name
		instName.strip()
		return self.cNNFeatures
#		return cNNInputFeatures[instName] 

	def setCNNFeatures(self, features):
		self.cNNFeatures = np.array(features)

	def getFeatures(self,kind):
		"""""""""""""""""""""""""""""""""""""""""
		Function to find the complete dataset file path (.../arch/arch_1/arch_1_rgb.log) 
		where the visual feaures are stored, read the features from the file, and return
	
         	Args: Instance class instance, type of features(rgb, shape, or object)
         	Returns: feature set
        """""""""""""""""""""""""""""""""""""""""     	
		instName = self.name
		if cumulativeFeatures == 'yes':
		    kDesc = self.getKernelDescriptorFeatures(kind)
		    cDesc = self.getCNNFeatures()
		    mX = max(len(kDesc), len(cDesc))
		    features = []
		    for ind in range(mX):
		         nFeature = list(kDesc[ind % len(kDesc)])
		         nFeature.extend(list(cDesc[ind % len(cDesc)]))
		         features.append(nFeature)
		    return features
		      
		if inputFeatures == 'kernelDescriptors' :
		    return self.getKernelDescriptorFeatures(kind)
		elif inputFeatures == 'cNNFeatures':
		    return self.getCNNFeatures()
        
	def addNegatives(self, negs):
		"""""""""""""""""""""""""""""""""""""""""
		Function to add negative instances
	
         	Args: Instance class instance, array of negative instances 
         	Returns: None
        """""""""""""""""""""""""""""""""""""""""
		self.negs = np.append(self.negs,negs)
		self.negs = list(set(self.negs))

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
#      self.negInstances = np.unique(np.append(self.negInstances,instName))
      self.negInstances = np.append(self.negInstances,instName)

   def getNegatives(self):
      """""""""""""""""""""""""""""""""""""""""
		Function to get all negative instances of this token (ex, "red")
	
         	Args: token class instance
         	Returns: array of negative instances (ex: arch/arch_1, ..)
      """""""""""""""""""""""""""""""""""""""""
      return np.sort(self.negInstances)

   def clearNegatives(self):
      self.negInstances = np.array([])


   def getTrainFiles(self,insts,kind,tests):
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
      if negSampling == 'negativeSampling':
           self.negInstances = np.array([], dtype='object')
           poss = list(set(list(self.posInstances)))
           negInsts = [inst for inst in list(insts.keys()) if inst not in tests if inst not in poss]
           self.negInstances = np.array(negInsts)
      if self.posInstances.shape[0] == 0 or self.negInstances.shape[0] == 0 :
         return (features,y)
      if self.posInstances.shape[0] > 0 :
        features = np.vstack([instances[inst][0].getFeatures(kind) for inst in self.posInstances])
      if self.negInstances.shape[0] > 0:
        negFeatures = np.vstack([instances[inst][0].getFeatures(kind) for inst in self.negInstances])
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
          num = inst.replace(cat + "_",'')
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

      for cat in sorted(cats.keys()):
         obj = cats[cat]
         tests = np.append(tests,obj[0].chooseOneInstance())
      tests = sorted(tests)
      return tests
      
   def getAllTokens(self, nDf, ):
       global allInstTokens
       instances = nDf.to_dict()
       df = read_table(self.annotationFile, sep=',',  header=None)  
       for column in df.values:
           ds = column[0]
           dsTokens = column[1].split(" ")
           dsTokens = list(filter(None, dsTokens))
           for tkn in dsTokens:
             allInstTokens.setdefault(ds,[]).append(tkn)

       
   def ablationStudyPreparations(self,cDf,nDf,tests):
      global allInstsForAblation
      instances = nDf.to_dict()
      categories = cDf.to_dict()
      global allInstDicts
      allInstDicts = instances
      if ablationPercentage == 100:
         allInstsForAblation = list(instances.keys())
         return (cDf,nDf)
      avlInsts = [inst for inst in list(instances.keys()) if inst not in tests]
      samplesToPick = int(float(len(avlInsts)) * float(ablationPercentage) / float(100))
      indicesToPick = random.sample(range(len(avlInsts)),samplesToPick)
      avlInsts = [inst for i,inst in enumerate(avlInsts) if i in indicesToPick]
      avlInsts.extend(tests)
      allInstsForAblation = avlInsts
      cats = {}
      insts = {}
      catNums = {}
      for instName in avlInsts:
         (cat,inst) = instName.split("/")
         num = inst.replace(cat + "_",'')
         if cat not in cats.keys():
            cats[cat] = categories[cat][0]
         catNums.setdefault(cat, []).append(num)
         if instName not in insts.keys():
          insts[instName] = instances[instName][0]
      for cat in cats.keys():
         cats[cat].clearAndUpdateCategoryInstances(catNums[cat])
      instDf = pd.DataFrame(insts,index=[0])
      catDf =  pd.DataFrame(cats,index=[0])
      return (catDf,instDf)
      
   def getDataSet(self,cDf,nDf,tests,tIndices):
      """""""""""""""""""""""""""""""""""""""""
        Function to add amazon mechanical turk description file, 
        find all tokens, find positive and negative instances for all tokens

             Args:  dataset instance, array of Category class instances, 
             	array of Instance class instances, array of instance names to test, 
             	file name for logging
             Returns:  array of Token class instances
      """""""""""""""""""""""""""""""""""""""""     	   
      global allInstsForAblation
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
        dsTokens = column[1].split(" ")
        dsTokens = list(filter(None, dsTokens))
        if ds not in allInstsForAblation:
          continue
        if ds in docs.keys():
           sent = docs[ds]
           sent += " " + column[1]
           docs[ds] = sent
        else:
           docs[ds] = column[1]
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

      negSelection = NegSampleSelection(docs)
      negExamples = negSelection.generateNegatives()
      for ds in negExamples.keys():
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
      	 	instances[ds][0].addNegatives(negatives)	
      """ find negative instances for all tokens.
      Instances which has cosine angle greater than 80 in vector space consider as negative sample"""
      for tk in tokenDf.keys():
         poss = list(set(tokenDf[tk].getPositives()))
         negs = []
         for ds in poss:
                  negatives = instances[ds][0].getNegatives()
                  negatives = [xx for xx in negatives if xx not in poss]
                  negs.extend(negatives)
                  
         negsPart = negs
         tokenDf[tk].extendNegatives(negsPart)
      return tks   

              
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
  for token in sorted(tokenDict.keys()):
#  for token in ['arch']:
        objTkn = tokenDict[token][0]
     
#     for kind in ['rgb']: 
        (features,y) = objTkn.getTrainFiles(insts,kind,tests)
        (testFeatures,testY) = getTestFiles(insts,kind,tests,token)
        if len(features) == 0 :
            continue;
        yield (token,kind,features,y,testFeatures,testY)
        
def getPosNegUnlabeledTokens(insts1, tests, tTested, kind, includetestData=False,labelsForImages=True, onlyTestData=False, allTknInsts=allInstTokens):
       insts = insts1
       if allImgForVAE == 1 and ablPr < 100:
           insts = allInstDicts
       x_insts = [inst for inst in insts.keys() if inst not in tests]
       if includetestData:
        x_insts = [inst for inst in insts.keys()]
       if allImgForVAE == 1 and ablPr < 100:
          x_insts = [inst for inst in list(allInstDicts.keys()) if inst not in tests]
          if includetestData:
             x_insts = [inst for inst in list(allInstDicts.keys())]
       if onlyTestData == True:
       	   x_insts = [inst for inst in insts.keys() if inst in tests]             
       posLabels = []
       negLabels = []
       unLabels = []
       
       for inst in x_insts:
        instImages = insts[inst][0].getImages()
        instFeatures = insts[inst][0].getKernelDescriptorFeatures(kind)
#         posTkns = sorted([tkn for tkn in list(set(insts[inst][0].getTokens())) if tkn in tTested])
#         negTkns = sorted(list(set([tkn for negInst in insts[inst][0].getNegatives() for tkn in insts[negInst][0].getTokens() if tkn not in posTkns if tkn in tTested])))
#        if allImgForVAE == 1 and ablPr < 100:
        posTkns = sorted([tkn for tkn in list(set(allTknInsts[inst])) if tkn in tTested])
        negTkns = sorted(list(set([tkn for negInst in insts[inst][0].getNegatives() for tkn in allTknInsts[negInst] if tkn not in posTkns if tkn in tTested])))
#        for i in range(instFeatures.shape[0]):
        unTokens = sorted([tkn for tkn in list(set(tTested)) if tkn not in posTkns if tkn not in negTkns])
#         print(inst,"--->",len(unTokens), "====>", unTokens)
        if labelsForImages == True:
         for i in range(len(instImages)):
                posLabels.append(posTkns)
                negLabels.append(negTkns)
                unLabels.append(unTokens)
        else:
         for i in range(instFeatures.shape[0]):
                posLabels.append(posTkns)
                negLabels.append(negTkns)
                unLabels.append(unTokens)

#        print("Values: ", [len(unLabels[i]) for i in range(len(unLabels))])       
#        print("Max: ", np.max([len(unLabels[i]) for i in range(len(unLabels))]))       
#        print("Min: ", np.min([len(unLabels[i]) for i in range(len(unLabels))]))                  
#        
#        print("Mean: ", np.mean([len(unLabels[i]) for i in range(len(unLabels))]))           

       posNegUnLabelBinaryVector = []                
       mlb = MultiLabelBinarizer(classes=tTested)
       labels = mlb.fit_transform(posLabels)
       ulabels = mlb.fit_transform(unLabels) * -1
#       posNegUnLabelBinaryVector = labels + ulabels
       posNegUnLabelBinaryVector = labels 
       return posLabels, negLabels, unLabels, posNegUnLabelBinaryVector
       
       
def getImagesAndFeatures(insts1,tests,tTested, kind,includetestData=False, labelsForImages=True, onlyTestData=False, allTknInsts=allInstTokens):
       insts = insts1
       if allImgForVAE == 'yes' and ablationPercentage < 100:
           insts = allInstDicts
#       x_insts = [inst for inst in insts.keys() if inst not in tests]
       x_insts = [inst for inst in insts.keys() if inst not in tests]
       if includetestData:
       	x_insts = [inst for inst in insts.keys()]
       if allImgForVAE == 'yes' and ablationPercentage < 100:
          x_insts = [inst for inst in list(allInstDicts.keys()) if inst not in tests]
          if includetestData:
             x_insts = [inst for inst in list(allInstDicts.keys())]

       if onlyTestData == True:
       	   x_insts = [inst for inst in insts.keys() if inst in tests]
       xImages = []
       labels = []
       trainFeatures = []

       for inst in x_insts:
       	instImages = insts[inst][0].getImages()
       	xImages.extend(instImages)
       	instFeatures = insts[inst][0].getKernelDescriptorFeatures(kind)
       	for i in range(instFeatures.shape[0]):
       		trainFeatures.append(instFeatures[i])
#         posTkns = sorted([tkn for tkn in list(set(insts[inst][0].getTokens())) if tkn in tTested])
#         if allImgForVAE == 1 and ablationPercentage < 100:
        posTkns = sorted([tkn for tkn in list(set(allTknInsts[inst])) if tkn in tTested])
#        for i in range(instFeatures.shape[0]):
        if labelsForImages == True:
         for i in range(len(instImages)):
         	labels.append(posTkns)
        else:
         for i in range(instFeatures.shape[0]):
         	labels.append(posTkns)          			        	
       return  xImages, trainFeatures, labels

def callML(resultDir,insts,tkns,tests):
  global modelNN, mlbNN
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
  
  tokenDict = tkns.to_dict()
  tobeTestedTokens = sorted(tokenDict.keys())
  tobeTestedTokens = sorted(list(set(tobeTestedTokens)))
  X_features = [] 
  vaeFeatures = []
  probabilities = []
  vaeMultilabelIndices = []
  if execType == 'vae':
     X_features, vaeFeatures, probabilities = VAE.getVAEFeatures(configParams, insts, tkns, tests, tobeTestedTokens, allInstTokens)
  """ fine tokens, type to test, train/test features and values """
  for (token,kind,X,Y,tX,tY) in findTrainTestFeatures(insts,tkns,tests):
   if token not in testTokens:
      testTokens.append(token)
   print ("Token : " + token + ", Kind : " + kind)
   X1 = []
   if execType == 'vae' and configParams['VAE']['testOption'] != 3:
       for xIn in X:
           ind = [i for i, xx in enumerate(X_features) if np.array_equal(xx, xIn)]
           X1.append(vaeFeatures[ind[0]])
       X = np.array(X1)

   print(X.shape)
   """ binary classifier Logisitc regression is used here """
   pipeline2_2 =   classifierModel(model=classModel, xDim = X.shape[1], yDim=1)


#   if execType == 'random' or execType == 'seq':
   if execType != 'cNNMultiLabel' and not(execType == 'vae' and configParams['VAE']['testOption'] == 3):
       classifierFit(pipeline2_2, X, Y, model=classModel)
   fldNames = np.array(['Token','Type'])  
   confD = {}
   ttX = []
   ttY = []
   testTT = []
   xTests = []
   confDict = {'Token' : token,'Type' : kind}
   """ testing all images category wise and saving the probabilitties in a Map 
   		for ex, for category, tomato, test all images (tomato image 1, tomato image 2...)"""
   for ii in range(len(tX)) :
      testX = tX[ii]
      testY = tY[ii]
      ttX.extend(testX)
      ttY.extend(testY)
      tt = tests[ii]
      ttNos = len(testY)
      if execType == 'cNNMultiLabel':
      	instImages = insts[tt][0].getImages()
      	xTests.extend(instImages)
      	ttNos = len(instImages)
      	 
      for ik in range(ttNos):
         fldNames = np.append(fldNames,str(ik) + "-" + tt)
         confD[str(ik) + "-" + tt] = str(fSet[tt])
         testTT.append(str(ik) + "-" + tt)

   
   if execType == 'vae' and configParams['VAE']['testOption'] != 3:
       X1 = []
       for xIn in  ttX:
           ind = [i for i, xx in enumerate(X_features) if np.array_equal(xx, xIn)]
           X1.append(vaeFeatures[ind[0]])
       ttX = np.array(X1)
   if execType == 'vae' and configParams['VAE']['testOption'] == 3:    
       for xIn in  ttX:
       	        ind = [i for i, xx in enumerate(X_features) if np.array_equal(xx, xIn)]
       	        vaeMultilabelIndices.append(ind)
       
   predY = []  
   tProbs = []
#   if execType == 'random' or execType == 'seq':
#         predY = pipeline2_2.predict(ttX)
   if execType != 'cNNMultiLabel' and not(execType == 'vae' and configParams['VAE']['testOption'] == 3):  
      probK = classifierPredict(pipeline2_2, ttX, model=classModel)
      tProbs = probK[:,1]
      for ik in range(len(tProbs)):
           confDict[testTT[ik]] = str(tProbs[ik])
           
   if headFlag == 0:
      headFlag = 1
      ### Execute only once
      
      if execType == 'cNNMultiLabel':
      	probabilities = mLabel.test(xTests, modelNN, mlbNN, nnName)
      """ saving the header of CSV file """
      confWriter = csv.DictWriter(confFile, fieldnames=fldNames)
      confWriter.writeheader()
      confWriter.writerow(confD)
      
   if execType == 'cNNMultiLabel'  or (execType == 'vae' and configParams['VAE']['testOption'] == 3):
   	classes = tobeTestedTokens
   	if execType == 'cNNMultiLabel' :
   		classes =  list(mlbNN.classes_)
   	if token in classes:
   		ind = classes.index(token)
   		tProbs  = probabilities[ind]
   		totalProbs  = len(testTT)
   		for ik in range(totalProbs):
   		   if execType == 'vae' and configParams['VAE']['testOption'] == 3 :
   		   	   confDict[testTT[ik]] = str(tProbs[vaeMultilabelIndices[ik]][0])
   		   else:
   		   	   confDict[testTT[ik]] = str(tProbs[ik])
   			
   		
   """ saving probabilities in CSV file """
   confWriter.writerow(confDict)

  confFile.close()


def execution(resultDir,ds,cDf,nDf,tests, totalTrainCases):
    resultDir1 = resultDir + "/NoOfDataPoints/" + str(numberOfDPs)
    os.system("mkdir -p " + resultDir1)
    tIndices = totalTrainCases
    if execType == 'random' or execType == 'seq':
      if execType == 'random':
        random.seed(4)
        random.shuffle(tIndices)
      if numberOfDPs < len(totalTrainCases):
        tIndices = tIndices[0:numberOfDPs]
    elif  execType ==   'al':
        tIndices = AL.getMeaningfulPoints(ds, nDf, tests, configParams)

    """ read amazon mechanical turk file, find all tokens
    get positive and negative instance for all tokens """
    tokens = ds.getDataSet(cDf,nDf,tests,tIndices)
    """ Train and run binary classifiers for all tokens, find the probabilities 
    	for the associations between all tokens and test instances, 
    	and log the probabilitties """
    callML(resultDir1,nDf,tokens,tests)
    
def getTotalTrainCases(anFile,tests,nDf,kind):
   totalTrainCases = 0
   df = read_table(anFile, sep=',',  header=None)
   cDz = df.values
   instances = nDf.to_dict()
#   totalTrainCases = [idx for idx,column in enumerate(cDz) if column[0] not in tests for inst in instances[column[0]][0].getFeatures(kind)]
   totalTrainCases = [idx for idx,column in enumerate(cDz) if column[0] not in tests]
   return totalTrainCases

if __name__== "__main__":
  global modelNN, mlbNN
  print ("Script START :: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
  fResName = ""
  
  os.system("mkdir -p " + resultDir)
  """ creating a Dataset class Instance with dataset path, amazon mechanical turk description file"""
  ds = DataSet(dsPath,preFile)
  """ find all categories and instances in the dataset """
  (cDf,nDf) = ds.findCategoryInstances()
  """ find all test instances. We are doing 4- fold cross validation """
  tests = ds.splitTestInstances(cDf)
  totalTrainCases = getTotalTrainCases(preFile,tests,nDf,kind)
  (cDf,nDf) = ds.ablationStudyPreparations(cDf,nDf,tests)
  ds.getAllTokens(nDf)
  
  if inputFeatures == 'cNNFeatures' or cumulativeFeatures == 'yes':
       
       allTkns = sorted(list(set([tkn for tkns in allInstTokens.values()  for tkn in tkns])))   
       xImages, trainFeatures, labels = getImagesAndFeatures(nDf,tests,allTkns,kind)
       model, mExtract, mlb, nnFeaturesAll = mLabel.classify(xImages,labels,nnName+'-glaML.model',nnName+'-mlb.pickle',nnName+'-loss.png', xImages,nnName, cNNTune)
       modelNN = model
       mlbNN = mlb
       for instName in list(nDf.keys()):
           	instImages = nDf[instName][0].getImages()
           	instData, image_dims = mLabel.preProcessImages(instImages, nnName)
           	cNNInputFeatures[instName] = mExtract.predict(instData)
           	nDf[instName][0].setCNNFeatures(cNNInputFeatures[instName])
           	
  execution(resultDir, ds, cDf, nDf, tests, totalTrainCases)
  print ("Script END :: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))