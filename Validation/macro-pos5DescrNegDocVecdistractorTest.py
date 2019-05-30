# This script tests and validates the performance of visual classifiers beyond category,
# by selecting right positive instances from the pool of positive and negative instance object images.
#
# Positive instances --> images of instances of which the visual classifier token is used at least 6 times to describe.
# Negative instances -- > the intersection of negative sampling and Doc2Vec negative instances of all positive instances
#

# Arguments: If your result directory is 'test/NoOfDataPoints/6000', then
# python macro-pos5DescrNegDocVecdistractorTest.py test/NoOfDataPoints/ 'rgb' 'rgb' <prefile>
# or
# python macro-pos5DescrNegDocVecdistractorTest.py test/NoOfDataPoints/ 'shape' 'shape' <prefile>
# or
# python macro-pos5DescrNegDocVecdistractorTest.py test/NoOfDataPoints/ 'object' 'object' <prefile>
# <prefile> is the path to the original preprocessed description file
#!/usr/bin/env python
import numpy as np
from pandas import DataFrame, read_table
import pandas as pd
import random
import util
from collections import Counter
import json
import sys
import csv
import os
import collections
import os.path
import math

#this is a constant of the number of times a token needs to appear in instance descriptions before the instance
#is counted as a positive example of that token
MIN_INSTS = 5
NEG_SAMPLE_PORTION = 0.25
#These are global variables that will be used later
posInsts = {}
negInsts = {}
objInstances = {}
objNames = {}
classifierProbs = {}

argvs = sys.argv
if len(argvs) == 1:
	exit(0)

fld = str(argvs[1])

argProb = 0.50
tID = ""

if len(argvs) > 2:
	tID =  str(argvs[2])

cID = ""
if len(argvs) > 3:
   cID = str(argvs[3])


preFile = "../6k_lemmatized_72instances_mechanicalturk_description.conf"
if len(argvs) > 4:
   preFile = str(argvs[4])

if len(argvs) > 5 and "train" in argvs:
   confFile = "groundTruthPredictionTrain.csv"
else:
   confFile = "groundTruthPrediction.csv"

posName = "posInstances.conf"
negName = "negInstances.conf"

pos = fld + "/evalHelpFiles/" + tID + "posInstances.conf"
neg = fld + "/evalHelpFiles/" + tID + "negInstances.conf"

####TFIDF MeaningfulWords - Above 50.0####

#These are the hardcoded vocabulary words
"""
if language == "spanish":
   rgbWords = ['amarillo','azul','morado','negro', 'isyellow','verde','marron','naranja','blanco','rojo' ]
   meaningfulWords = rgbWords
else:
   meaningfulWords = ['cylinder', 'apple', 'yellow', 'carrot', 'lime', 'blue', 'lemon', 'purple', 'orange', 'banana', 'red', 'cube', 'triangle', 'corn', 'triangular', 'cucumber', 'half', 'cabbage', 'ear', 'tomato', 'potato', 'rectangular', 'cob', 'green', 'eggplant']
   rgbWords  = ['yellow','blue','purple', 'orange','red','green']
shapeWords  = ['cylinder','cube', 'triangle','triangular','rectangular']
objWords = ['cylinder', 'apple','carrot', 'lime','lemon','orange', 'banana','cube', 'triangle', 'corn','cucumber', 'half', 'cabbage', 'ear', 'tomato', 'potato', 'cob','eggplant']
objWords = list(set(objWords) - set(['ear','half']))
"""

#No tokens have appeared yet
predfName = fld + "/"+confFile
predfileName = fld

def getPosNegInstances():
	"""
	This function reads in the posInstances.csv and negInstances.csv files
	The values are stored in the posInsts and negInsts global variables
	"""
	resDictFile = []
	with open(pos, 'rU') as csvfile:
		readFile = csv.DictReader(csvfile)
		for row in readFile:
			resDictFile.append(row)
			color = row['token']
			objs = row['objects']
			insts = objs.split("-")
			newInsts = []
			for instance in insts:
				if instance in objInstances.keys():
					newInsts.append(instance)
					posInsts[color] = newInsts

	resDictFile = []
	with open(neg, 'rU') as csvfile:
	  readFile = csv.DictReader(csvfile)
	  for row in readFile:
		  resDictFile.append(row)
		  color = row['token']
		  objs = row['objects']
		  insts = objs.split("-")
		  newInsts = []
		  for instance in insts:
			  if instance in objInstances.keys():
				  newInsts.append(instance)
				  negInsts[color] = newInsts


def getTestInstancesAndClassifiers(fName):
	"""
	This function opens the groundTruthPrediction.csv file. This file has
	the tokens, the "ground truth", and the probability of each test instance when
	the color token for that classifier is applied to it.

	"""
	global objInstances,objNames,classifierProbs
	head = 0
	#This holds the names of all instances of each type of object
	objInstances = {}

	#This is a dictionary that holds the "ground truth" for each of the test instances
	objNames = {}

	#This gives the token and the probability of each test instance matching that token (by applying the classifier
	#trained for that token to the test image)
	classifierProbs = {}
	with open(fName) as csvfile:
		readFile = csv.DictReader(csvfile)
		for row in readFile:
			if head == 0:
				temp = row.keys()

				temp.remove('Type')
				temp.remove('Token')
				for inst in temp:
					ar = inst.split("-")
					if ar[1] in objInstances.keys():
						objInstances[ar[1]].append(inst)
					else :
						objInstances[ar[1]] = [inst]
						objNames[ar[1]] = row[inst]
			else :
                                if cID == "":
					#if row['Token'] in posInsts.keys():
					if row['Token'] in classifierProbs.keys():
						classifierProbs[row['Token']].append(row)
					else:
						classifierProbs[row['Token']] = [row]
				else:
					if row['Type'] == cID:
						if row['Token'] in classifierProbs.keys():
							classifierProbs[row['Token']].append(row)
						else:
							classifierProbs[row['Token']] = [row]

			head = head + 1

def writePosNeg():
	 """
	 This function actually creates the positive and negative example files that
	 are used later. They only bother to find the examples for the "meaningful" words
	 """
	 testObjs = objInstances.keys()
	 #get the raw desriptions of the test objects from the preprocessed file
	 descObjs = util.getDocsForTest(testObjs,preFile)

	 #Take the descriptions and turn them into a list of words.
	 #This list contains all of the words used in all descriptions of this instance
	 objTokens = util.sentenceToWordDicts(descObjs)

	 tknsGlobal = set()
	 posTokens = {}
	 negSampleTokens = {}
	 mostImpTokens = {}

	 #If there are no test tokens, return 1
	 if len(objTokens.keys()) == 0:
	   return 1

	 #loop over the test instances. The key is he instance and the value
	 #is the list of all words in the instance's descriptions
	 inst_token_count_dict = {}#TEMP to count the number of times a token is used with each inst
	 for (key,value) in objTokens.items():
	    #This counts the number of times each token was used
	    cValue = Counter(value)
	    mostImpTokens[key] = []
	    inst_token_count_dict[key] = {}
	    for (k1,v1) in cValue.items():
		     k1 = k1.replace("\r","").replace("\n","").replace("\t","")
		     inst_token_count_dict[key][k1] = v1
		     #If the token was used more than 10 times, it is important
		     if v1 > 10:
			mostImpTokens[key].append(k1)
		     if v1 >= MIN_INSTS:
			#This is a spot where we only are looking at the
			#words that have been identified as "meaningful"
			#if k1 in meaningfulWords:
			tknsGlobal.add(k1)
			#if the token was a "meaningful word" with a high count,
			#then the instance key is a positive example of that token.
			if k1 in posTokens.keys():
			     kk1 = posTokens[k1]
			     kk1.append(key)
			     posTokens[k1] = kk1
			else:
			     posTokens[k1] = [key]
	 inst_token_count_frame = pd.DataFrame(inst_token_count_dict).fillna(0)

	 inst_token_count_frame.to_csv(fld + "/token_instance_counts.csv")
	 posTokens = collections.OrderedDict(sorted(posTokens.items()))
	 #Output a file which maps tokens to the test instances that are positive examples of them
	 if not os.path.exists(fld + "/evalHelpFiles/"):
	     os.mkdir(fld + "/evalHelpFiles/")
	 f = open(fld + "/evalHelpFiles/" + posName, "w")
	 title = "token,objects\n"
	 f.write(title)
	 for k,v in posTokens.items():
	    ll  = str(k) + ","
	    ll += "-".join(v)
	    ll += "\n"
	    f.write(ll)
	 f.close()

	 #loop over the tokens that were found by the positive test. Now we want to
	 #get the negative examples. First, get the instances for each positive token where
	 #that token never appears in the descriptions.
	 for kTkn in posTokens.keys():
	    negSampleTokens[kTkn] = []
	    for (key,value) in objTokens.items():
	       if kTkn not in value:
		  negSampleTokens[kTkn].append(key)
	 negTokens = {}
	 #Now we get the doc2vec versions of the description strings for the instances
	 #we just identified
	 negsD = util.doc2Vec(descObjs)

	 for kTkn in posTokens.keys():
	   negTokens[kTkn] = negSampleTokens[kTkn]
	   posV = posTokens[kTkn]
           #This dictionary keeps track of how overal negative an instance is
           sum_negative_dict = {}
	   #Get the instances that are both far away with doc2vec and also dont have descriptions that use that token
	   for v in posV:

              #get the negative instances and their distances. Add it to the overal dict
              #only consider instances that hace not seen the token in their descriptions
              negDocVec = negsD[v]

              for (negInst,distance) in negDocVec:
                 if negInst in negSampleTokens[kTkn]:
                    if negInst in sum_negative_dict:
                       sum_negative_dict[negInst] += distance
                    else:
                       sum_negative_dict[negInst] = distance

	      #negTokens[kTkn] = list(set(negTokens[kTkn]).intersection(set(negDocVec)))
              #negTokens[kTkn] = list(set(negSampleTokens[kTkn]).intersection(set(negDocVec)) | set(negTokens[kTkn]))
           #take the top N negative examples by the weighted votes from all positive instances
           sum_negs_sorted = sorted(sum_negative_dict.iteritems(),key = lambda x: x[1], reverse=True)
           num_to_choose = int(math.ceil(float(len(sum_negs_sorted))*NEG_SAMPLE_PORTION))
           #num_to_choose = min(num_to_choose,2*len(posV))
           negTokens[kTkn] = [negInst for negInst, negVal in sum_negs_sorted[:num_to_choose]]

         #write the negative tokens to the file
	 negTokens = collections.OrderedDict(sorted(negTokens.items()))
	 f = open(fld + "/evalHelpFiles/" + negName, "w")
	 f.write(title)
	 for k,v in negTokens.items():
	    ll  = str(k) + ","
	    ll += "-".join(v)
	    ll += "\n"
	    f.write(ll)
	 f.close()

	 #It grabs the positive and negative tokens of just those words with positive instances
	 kWord = ["rgb","shape","object"]
	 for wd in kWord:
	   f = open(fld + "/evalHelpFiles/" + wd + posName, "w")
	   f1 = open(fld + "/evalHelpFiles/" +wd + negName, "w")
	   sWords = []
	   f.write(title)
	   f1.write(title)
	   for k,v in posTokens.items():
	     if len(v) > 0:
	       ll  = str(k) + ","
	       ll += "-".join(v)
	       ll += "\n"
	       f.write(ll)
	     v = negTokens[k]
	     if len(v) > 0:
	       ll  = str(k) + ","
	       ll += "-".join(v)
	       ll += "\n"
	       f1.write(ll)
	   f.close()
	   f1.close()
	 return 0

def getTestImages(testPosInsts,testNegInsts,posNo,negNo):
	"""
	This function takes in lists of instance names like "cuboid/cuboid_2"
	for the positive and negative examples of a token. It also takes in
	randomly chosen values for the number of positive and negative instances to
	select. It then selects the given number of instances from the lists.
	"""
	posId = []
	negId = []
	testInstances = {}
	relevantInst = []
	if (len(testPosInsts) > 0) or (len(testNegInsts) > 0):
		#This makes a list of indices to select moded by how many
		#elements are actually available
		if len(testPosInsts) > 0:
			posId = [(i + 1) %  len(testPosInsts) for i in range(posNo)]
		if len(testNegInsts) > 0:
			negId = [(inn + 1) % len(testNegInsts) for inn in range(negNo)]

		for id in posId:
			tmp = testPosInsts[id]
			#This gets us the list of images from a particular positive example (each has 5)
			imgs = objInstances[tmp]

			#For each of the 4 instances of each object, there are also some number of images. Choose one of these
			#images randomly
			p1 = random.sample(range(len(imgs)), k=1)
			testInstances[imgs[p1[0]]] = tmp + " (" + objNames[tmp] + ")"
			relevantInst.append(imgs[p1[0]])

		for id in negId:
			tmp = testNegInsts[id]
			imgs = objInstances[tmp]
			p1 = random.sample(range(len(imgs)), k=1)
			testInstances[imgs[p1[0]]] = tmp + " (" + objNames[tmp] + ")"
	return (relevantInst,testInstances)

def selectCorrectImage(c,testInstances):
	"""
	This function goes over the classifier likelihoods for a token, and
	selects the images that have higher than argmax score. Basically, the
	classifier says that the image is at least argmax likely to be a positive
	example of the token
	"""
	probs = classifierProbs[c]
	#argProb is a fixed value (right now 0.5)
	argMax = argProb
	selInst = []
	for ik in range(len(probs)):
		probInst = probs[ik]
		for inst in testInstances.keys():
			if float(probInst[inst]) > argMax:
				#  argMax = float(probs[inst])
				selInst.append(inst)
	return list(set(selInst))

def getMatchNumbers(relevantInst,selInst,testInstances, token, token_confusion_dict):
      """
	This function takes the list of the instances that were
	selected to look at (positive)
	and the instances the classifier deemed
	positive.
	the TP/FP/TN/FN scores
      """

      tNo = float(len(testInstances))
      tP = 0.0
      fN = 0.0
      fP = 0.0
      tps = []
      #the true positives are the ones chosen by the classifier and also chosen as relevant positive examples
      if len(relevantInst) > 0:
            tps = list(set(relevantInst).intersection(set(selInst)))
            for true_pos in tps:
                  if true_pos not in token_confusion_dict["TP"]:
                        token_confusion_dict["TP"][true_pos] = 1
                  else:
                        token_confusion_dict["TP"][true_pos] += 1
            #the stuff that was selected but not actually positive
            fps = list(set(selInst).difference(set(relevantInst)))
            for false_pos in fps:
                  if false_pos not in token_confusion_dict["FP"]:
                        token_confusion_dict["FP"][false_pos] = 1
                  else:
                        token_confusion_dict["FP"][false_pos] += 1

            #the stuff that was positive but not selected
            fns = list(set(relevantInst).difference(set(selInst)))
            for false_negs in fns:
                  if false_negs not in token_confusion_dict["FN"]:
                        token_confusion_dict["FN"][false_negs] = 1
                  else:
                        token_confusion_dict["FN"][false_negs] += 1

            #the stuff that was not selected and was negative
            tns = set(testInstances).difference(set(selInst).union(set(relevantInst)))
            for true_negs in tns:
                  if true_negs not in token_confusion_dict["TN"]:
                        token_confusion_dict["TN"][true_negs] = 1
                  else:
                        token_confusion_dict["TN"][true_negs] += 1

      tP = float(len(tps))
      #The false positives are the ones chosen as selInst but not actually positive
      fP = float(len(selInst) - tP)
      fN = float(len(relevantInst) - tP)
      tN = tNo - tP - fN - fP
      return (tP,fN,fP,tN)

def getStats(tP,fN,fP,tN):
	"""
	This function finds accuracy, precision, recall, and f1-score
	"""
	acc = (tP + tN)/(tP + fN + fP + tN)
	if (tP + fP) > 0.0:
		prec = tP / (tP + fP)
	else:
		prec = 0.0
	if (tP + fN) > 0.0:
		rec = tP / (tP + fN)
	else:
		rec = 0.0
	f1s = 0.0
	if (prec + rec) != 0.0:
		f1s = 2.0 * prec * rec / (prec + rec)
	return (acc,prec,rec,f1s)


rootDir = fld
resultFolder = fld

#open the result file and write the header
resFileName = resultFolder + tID + "_" + str(cID) + "_overall-learnPerformance.csv"
perfFile1 = open(resFileName,'w')
fieldnames = np.array(['Test Batch'])
fieldnames = np.append(fieldnames,['True Positive','True Negative','False Positive','False Negative'])
fieldnames = np.append(fieldnames,['Accuracy','Precision','Recall','F1-Score'])
writer1 = csv.DictWriter(perfFile1, fieldnames=fieldnames)
writer1.writeheader()

noIndArs = []
accArs = []
f1sArs = []
precArs = []
recArs = []
#for fNo in np.arange(100,2000,100):
#for fNo in np.arange(10,200,10):
#for fNo1 in range(100):

#This list was made for when testing the different number of samples. Right now, this is just the 6000 folder
fFldrs = []
for o in os.listdir(rootDir):
  if os.path.isdir(os.path.join(rootDir,o)):
      fFldrs.append(int(o))
fFldrs = np.sort(fFldrs)
for fNo in fFldrs:
    fName1 = predfileName + "/" + str(fNo)

    resultFolder = fName1
    fld = resultFolder

    #fname is the ground truth prediction file that is made during training
    fName = fName1 + "/" + confFile
    predfName = fName

    #These are the files that give positive and negative instances for each token. They will be populaed and
    #used later for evaluation
    pos = fName1 + "/evalHelpFiles/" + tID + "posInstances.conf"
    neg = fName1 + "/evalHelpFiles/" + tID + "negInstances.conf"

    #open another results file and write the header
    resultFileName = resultFolder + "/" + str(fNo) + "-" + str(tID) + "_" + str(cID) + "_learnPerformance.csv"
    perfFile = open(resultFileName,'w')
    fieldnames = np.array(['Classifier','Test Object Images','Ground Truth','Selected by Classifier'])
    fieldnames = np.append(fieldnames,['True Positive','True Negative','False Positive','False Negative'])
    fieldnames = np.append(fieldnames,['Accuracy','Precision','Recall','F1-Score'])
    writer = csv.DictWriter(perfFile, fieldnames=fieldnames)
    writer.writeheader()

    #call the function that opens the groundTruthPrediction file to get the
    #probabilities of all instances for the token classifiers
    getTestInstancesAndClassifiers(fName)

    #call the function that populates the positive and negative test instances files
    ret = writePosNeg()

    #There were no test tokens!
    if ret == 1:
     noIndArs.append(str(fNo))
     accArs.append(str(0))
     f1sArs.append(str(0))
     precArs.append(str(0))
     recArs.append(str(0))
     continue

    #this reads in the positive and negative instance files that were just made
    getPosNegInstances()

    #get the list of tokens we are looking at which we have probabilities for and have positive/negative instances
    testTokens = list(set(posInsts.keys()).union(set(negInsts.keys())))
    #NOTE: this would mean we would only test on the tokens seen in training
    #testTokens = list(set(testTokens).intersection(set(classifierProbs.keys())))


    accFldr= []
    f1sFldr = []
    precFldr = []
    recFldr= []


    #loop over the tokens that we do have examples for
    print "token,accuracy,precision,recall,f1,num_positive_instances,num_negative_instances"

    for c in testTokens:
	  #print "number of testTokens: ",len(testTokens)
	  dictRes = {'Classifier' : " "}
	  writer.writerow(dictRes)
          testPosInsts = []
          testNegInsts = []
          accT = 0.0
          f1T = 0.0
          precT = 0.0
          recT = 0.0
	  #print "length of posInsts.keys: ",len(posInsts.keys())
	  if c in posInsts.keys():
		if len(posInsts[c]) > 0:
			#objInstances has the set of test instances (as written in the ground truth test file)
			#We only want to test over the intersection of the test instances and the positive instances
			testPosInsts = list(set(posInsts[c]).intersection(set(objInstances.keys())))
	  #print "length of negInsts.keys: ",len(negInsts.keys())
          if c in negInsts.keys():
		if len(negInsts[c]) > 0:
			testNegInsts = list(set(negInsts[c]).intersection(set(objInstances.keys())))

	  #After subsetting by the test instances, see if we still have any to test
          if len(testPosInsts) > 0 or len(testNegInsts) > 0:

                  #If this token was not learned in training, it scores 0 in testing
                  if c not in list(classifierProbs.keys()):
                     print str(c)+",0,0,0,0,"+str(len(testPosInsts))+","+str(len(testNegInsts))
                     accFldr.append(0.0)
                     f1sFldr.append(0.0)
                     precFldr.append(0.0)
                     recFldr.append(0.0)
                     continue


                  accTkn = []
                  f1sTkn = []
                  precTkn = []
                  recTkn = []
		  #repeat this test 10 times and average the results
		  token_confusion_dict = {"TP":{},"FP":{},"TN":{},"FN":{}}

		  for tms in range(10):
			  #choose a random number between 0,1,2 for the number of positive instances to look at
			  #TODO: why these exact values??
			  posNo = random.sample(range(3), k=1)
			  #choose a random number between 4,5,6
                          totNo = random.sample([4,5,6], k=1)
			  #The remaining difference is the number of test instances
			  negNo = totNo[0] - posNo[0] - 1

			  #send the list of positive and negative instances, as well as those two random numbers
			  #get a list of specific instances to iterate over, and subset testImages by the exact instances
			  (relevantInst,testInstances) = getTestImages(testPosInsts,testNegInsts,posNo[0] + 1, negNo)

			  #get the list of test images that are selected by the token classifiers to be positive instances
			  selInst = selectCorrectImage(c,testInstances)

			  #get the TP,FP,TN,FN scores for this batch of test instances and token
			  #calculate the stats from this
			  (tP,fN,fP,tN) = getMatchNumbers(relevantInst,selInst,testInstances, c, token_confusion_dict)
			  (acc,prec,rec,f1s) = getStats(tP,fN,fP,tN)

			  #Write the results in the csv file
			  tmpObj = ""
			  for v in testInstances.values():
				  tmpObj += str(v) + "      "
			  relInsts = ""
			  for ik in relevantInst:
				  relInsts += str(testInstances[ik]) + " "
			  selInsts = ""

			  for ik in selInst:
				  selInsts += str(testInstances[ik]) + " "
			  dictRes = {'Classifier' : str(c),'Test Object Images' : tmpObj, 'Ground Truth' : str(relInsts)}
			  dictRes.update({'Selected by Classifier' : str(selInsts), 'Accuracy' : str(acc),'Precision' : str(prec) ,'Recall' : str(rec),'F1-Score' : str(f1s)})
			  dictRes.update({'True Positive' : str(tP),'True Negative' : str(tN) ,'False Positive' : str(fP),'False Negative' : str(fN)})

			  writer.writerow(dictRes)
                          accTkn.append(acc)
                          f1sTkn.append(f1s)
                          precTkn.append(prec)
                          recTkn.append(rec)
		  #Get the average scores for the 10 runs
                  accT = np.mean(accTkn)
                  f1T = np.mean(f1sTkn)
                  precT = np.mean(precTkn)
                  recT = np.mean(recTkn)

          print str(c)+","+str(accT)+","+str(precT)+","+str(recT)+","+str(f1T)+","+str(len(testPosInsts))+","+str(len(testNegInsts))

	  #write the results to the overal file
          dictRes = {'Classifier' : 'Total - ' + str(c),'Accuracy' : str(accT),'Precision' : str(precT) ,'Recall' : str(recT),'F1-Score' : str(f1T)}
          writer.writerow(dictRes)
          accFldr.append(accT)
          f1sFldr.append(f1T)
          precFldr.append(precT)
          recFldr.append(recT)
    #The overal score is the average score for the selected tokens
    if len(accFldr) > 0:
      acc = np.mean(accFldr)
      prec = np.mean(precFldr)
      rec = np.mean(recFldr)
      f1s = np.mean(f1sFldr)
    else:
      acc = 0.0
      prec = 0.0
      rec = 0.0
      f1s = 0.0

    dictRes = {'Test Batch' : str(fNo)}
    dictRes.update({'Accuracy' : str(acc),'Precision' : str(prec) ,'Recall' : str(rec),'F1-Score' : str(f1s)})
    writer1.writerow(dictRes)
#    print dictRes
    perfFile.close()
    noIndArs.append(str(fNo))
    accArs.append(str(acc))
    f1sArs.append(str(f1s))
    precArs.append(str(prec))
    recArs.append(str(rec))

#dictRes = {'Test Batch' : " "}
#writer1.writerow(dictRes)
#(acc,prec,rec,f1s) = getStats(otP,ofN,ofP,otN)
#dictRes = {'Test Batch' : 'Total'}
#dictRes.update({'Accuracy' : str(acc),'Precision' : str(prec) ,'Recall' : str(rec),'F1-Score' : str(f1s)})
#dictRes.update({'True Positive' : str(otP),'True Negative' : str(otN) ,'False Positive' : str(ofP),'False Negative' : str(ofN)})
#writer1.writerow(dictRes)

#dRes = {'Test Batch' : 'Total'}
#dRes.update({'Accuracy' : str(acc),'Precision' : str(prec) ,'Recall' : str(rec),'F1-Score' : str(f1s)})

#print dRes
perfFile1.close()
print "Folder Number: ",
print ", ".join(noIndArs)
print "Accuracy: ",
print ", ".join(accArs)
print "F1-Score: ",
print ", ".join(f1sArs)
print "Precision: ",
print ", ".join(precArs)
print "Recall: ",
print ", ".join(recArs)
