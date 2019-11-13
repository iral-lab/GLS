#!/usr/bin/env python
import numpy as np
from pandas import DataFrame, read_table
from sklearn import mixture
import scipy.stats
import numpy.linalg as la
from itertools import product
from numpy import linalg as LA
import os
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import orth
import cLLML

dppPoints = 5
execType = 'alpool'
selectionType = 'max'
gmmPoints = 0
iteractionNo = 1
testInstanceID = 0
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

        print ("Co variance ", i , " ->",self.arCovs[i])

        dx = x - self.arMean[i]
        A =  la.inv(np.array(self.arCovs[i]))
        print (la.det(self.arCovs[i]))
        fE = (la.det(self.arCovs[i]) * 2.0 * np.pi) ** 0.5 
        gS = np.exp(-0.5 * np.dot(np.dot(dx,A),dx)) / fE
        return gS

   def em(self, data, nsteps = 10):

        k = self.noComps
        d = data.shape[1]
        n = len(data)

        for l in range(nsteps):
            print ("STEPS", l)
            # E step

            responses = np.zeros((k,n))
            for j in range(n):
                for i in range(k):
                    pdfVal = self.pdf(i,data[j])
                    responses[i,j] = self.priors[i] * pdfVal
            print ("RESPONSES START",responses.shape)
            print (responses)
            responses = responses / np.sum(responses,axis=0) # normalize the weights
            print ("2nd Normalized RESPONSESS",responses)
            print ("Responses -- DONE")
            # M step
            N = np.sum(responses,axis=1)
            print ("N",N, N.shape)
            for i in range(k):
                print ("I ", i , " in K ", k)
                mu = np.dot(responses[i,:],data) / N[i]
                print ("MU ",mu)
                sigma = np.zeros((d,d))
                print ("SPECIAL", d)
                for j in range(n):
#                   sigma += responses[i,j] * np.outer(data[j,:] - mu, data[j,:] - mu)
#                   sigma += responses[i,j] * np.dot(data[j,:] - mu, data[j,:] - mu)
                   ch = np.array(data[j,:] - mu)
                   ch = np.reshape(ch,(ch.shape[0],1))
                   sigma += responses[i,j] * np.dot(ch,ch.T)
#                    print data[j,:] - mu
#                    print np.dot(ch,ch.T)
#                    print responses[i,j]
#                    print "ADDING COMPONENT",responses[i,j] * np.dot(ch,ch.T)
# 
#                    print "NEW SIGMA FOR ",j,"-->",sigma
#                 print "SIGMA ", sigma
                sigma = sigma / N[i]
                print ("SIGMA NORM ", sigma)
                self.arMean[i] = mu
                self.arCovs[i] = sigma
#                det = np.fabs(la.det(sigma))
#                print "DETER ", det
#                factor = (2.0 * np.pi)**(data.shape[1] / 2.0) * (det)**(0.5)
#                print "FACTOR ", factor
#                self.arFactors[i] = factor
                self.priors[i] = N[i] / np.sum(N) # normalize the new priors
                print ("NEW PRIORS ", self.priors)
        print ("THIS STEP IS FINISHED")
  
###########################################################

class ALPoints:
   def __init__(self, path, anFile):
      self.dsPath = path
      self.annotationFile = anFile
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
      if execType == 'algmmdpp' and (data.shape[0] >= gmmPoints):
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
      qq = pdist(z, 'sqeuclidean')
      vecs,vals = self.decompose_kernel(L)
      dppIndices = self.sample_dpp(vals, vecs, k)
      return dppIndices
#######################################################################
   
   def generateALDPPTrainingInstances(self,nDf,tests,kind,totalNeeded):
      gmmFName = "GMMIndices/" + str(iteractionNo) + "-" + str(testInstanceID) + "-" + str(execType) + "-" + str(kind) + "-"
      os.system("mkdir -p GMMIndices")
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
         np.random.shuffle(ars)
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
      print (len(relDataSets),len(list(set(relDataSets))))
      indicesTobeTrained = []
      while(trainInstIndices > len(indicesTobeTrained)):
        for inst in relDataSets:
          if len(instIndices[inst]) > 0:
               ars = instIndices[inst]
               indicesTobeTrained.append(ars.pop(0))
               instIndices[inst] = ars
      strIndicesToBeTrained = [str(ind) for ind in indicesTobeTrained]
      lineIndices = " ".join(strIndicesToBeTrained)
      os.system("echo \'" + lineIndices + "\' >  " + gmmFName + "indicesGMMbased.log")
      print (len(indicesTobeTrained),len(list(set(indicesTobeTrained))))
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
      if execType == 'alpool' or execType == 'ald2VgmmPool':
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
          
   def getLangd2VecGMMIndicesfromDenistyPDFs(self,densityPDFs,dataInsts,dataInds,cDz):
          gMMClusters = {}
          for instInd,k in enumerate(dataInsts):
            myInd = np.argmax(densityPDFs[instInd])
            myVal = float(densityPDFs[instInd][myInd])
            if myInd in gMMClusters.keys():
              xx = gMMClusters[myInd]
              xx.update({instInd:myVal})
              gMMClusters[myInd] = xx
            else:
               gMMClusters[myInd] = {instInd:myVal}
          indicesTobeTrained = []
          if execType == 'ald2VgmmPool':
# or execType == 'ald2VgmmUncert':
             availNo = len(dataInsts)
             gMMClusters1 = {}
             for cNo in gMMClusters.keys():
                 cValues = gMMClusters[cNo]
#                 cValues = collections.OrderedDict(sorted(cValues.items(), reverse=True))
#                 cValues = sorted(cValues.iteritems(), key=lambda (k,v): (v,k),reverse=True)
                 cValues = sorted(map(lambda x, y: (y, x), cValues.keys(), cValues.values()), reverse=True)
                 gMMClusters1[cNo] = [k[1] for k in cValues]
             gMMClusters = gMMClusters1
             gMMInstances = []
             while len(gMMInstances) < len(dataInsts):
              for cNo in gMMClusters.keys():
               cValues = gMMClusters[cNo]
               if len(cValues) > 0:
                 dInd = cValues.pop(0)
                 gMMInstances.append(dataInds[dInd])
                 print (dataInds[dInd],cDz[dataInds[dInd]])
                 gMMClusters[cNo] = cValues
             return gMMInstances


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
                 cValues = sorted(map(lambda x, y: (y, x), cValues.keys(), cValues.values()), reverse=True)
                 gMMClusters1[cNo] = [k[1] for k in cValues]
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
                 tPnts  = sorted(map(lambda x, y: (y, x), tPnts.keys(), tPnts.values()), reverse=True)
                 utPnts = sorted(map(lambda x, y: (y, x), utPnts.keys(), utPnts.values()), reverse=True)
                 certainPoints[cNo] = [key[1] for key in tPnts]
                 uncertainPoints[cNo] = [key[1] for key in  utPnts]
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

   def generateLangd2VecGMMALTrainingInstances(self,nDf,tests,kind,totalNeeded):
      gmmFName = "GMMIndices/" + str(iteractionNo) + "-" + str(testInstanceID) + "-" + str(execType) + "-" + str(kind) + "-"
      os.system("mkdir -p GMMIndices")
      global noComponents
      df = read_table(self.annotationFile, sep=',',  header=None)
      cDz = df.values
      instances = nDf.to_dict()
      instIndices = {}
      instFiles = []
      instDescs = []
      for column in cDz:
          instFiles.append(column[0])
          instDescs.append(column[1])

      dataFeatures = {}
      dataInsts = []
      dataVecs = []
      data = []
      docs = {}
      dataInds = []
      for ind,instName in enumerate(instFiles):
        if instName not in tests:
           instName1 = instName + "_" + str(ind)
           instsData = instances[instName][0].getFeatures(kind)
           dataFeatures[instName1] = instsData[0]
           docs[instName1] = instDescs[ind].strip()
           dataInsts.append(instName1)
           dataInds.append(ind)

      negSelection = cLLML.NegSampleSelection(docs)
      dVec = negSelection.getDoc2Vec(vectorSize=100)
      for instName in dataInsts:
          d1 = dataFeatures[instName]
          d2 = dVec[instName]
          d1 = np.array(d1).reshape(1,len(d1))
          d2 = np.array(d2).reshape(1,len(d2))
          d11 =  preprocessing.normalize(d1,norm = 'l2',axis=1)
          d22 =  preprocessing.normalize(d2,norm = 'l2',axis=1)
          dt = np.append(d11,d22)
          data.append(dt)
      gmm = mixture.GaussianMixture(n_components=gmmPoints, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='random', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)
      gmm.fit(data)
      densityPDFs = {}
      for i in range(gmm.n_components):
        density = scipy.stats.multivariate_normal(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(data)
        for idx,dens in enumerate(density):
          densityPDFs.setdefault(idx, []).append(density[idx])
      indicesTobeTrained = []
      if selectionType == 'max':
            indicesTobeTrained = self.getLangd2VecGMMIndicesfromDenistyPDFs(densityPDFs,dataInsts,dataInds,cDz)
      strIndicesToBeTrained = [str(ind) for ind in indicesTobeTrained]
      lineIndices = " ".join(strIndicesToBeTrained)
      os.system("echo \'" + lineIndices + "\' >  " + gmmFName + "indicesGMMbased.log")
      if totalNeeded > len(indicesTobeTrained):
             return list(np.sort(indicesTobeTrained))
      if execType == 'ald2VgmmPool':
            return list(np.sort(indicesTobeTrained[0:totalNeeded]))

   def generateRandomInstances(self,nDf,tests,kind,totalNeeded,seedNumber):
      df = read_table(self.annotationFile, sep=',',  header=None)
      cDz = df.values
      instIndices = {}
      instFiles = []
      for column in cDz:
        if column[0] not in tests:
          instFiles.append(column[0])
      instFilesSet = list(set(instFiles))
      random.seed(seedNumber)
      random.shuffle(instFilesSet)
      indicesNeeded = []
      for instName in instFilesSet:
          inds = np.argwhere(np.array(instFiles)==instName)
          ars = [ind[0] for ind in inds]
          random.shuffle(ars)
          indicesNeeded.append(ars[0])
          instIndices[instName] = ars
      return indicesNeeded
      
   def generateALTrainingInstancesPool(self,nDf,tests,kind,totalNeeded):
      gmmFName = "GMMIndices/" + str(iteractionNo) + "-" + str(testInstanceID) + "-" + str(execType) + "-" + str(kind) + "-"
      os.system("mkdir -p GMMIndices")
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
         np.random.shuffle(ars)
         instIndices[instName] = ars

      data = []
      dataInsts = []
      for instName in instances.keys():
        if instName not in tests:
           instsData = instances[instName][0].getFeatures(kind)
           data.append(instsData[0])
           dataInsts.append(instName)

      data = np.array(data)
      learnInstances = {}
      for instName in list(set(dataInsts)):
          inds = np.argwhere(np.array(dataInsts)==instName)
          ars = [ind[0] for ind in inds]
          learnInstances[instName] = ars
      trainInstIndices = len(dataInsts)

      relPoints = trainInstIndices
      relDataSets = []
      if relPoints > 0:
#          gmm = GMM(noComps = gmmPoints,data = data)
#          gmm.em(data)
          gmm = mixture.GaussianMixture(n_components=gmmPoints, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)
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
          f = open(gmmFName + "DensityPDFsGMMBased.log","w")
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
          f = open(gmmFName + "predictProbGMMBased.log","w")
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
          os.system("echo \'" + lineIndices + "\' >  " + gmmFName + "indicesGMMbased.log")
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
#           print "<table>"  
#           print "<tr><td>Number of Components :: </td><td>" + str(gmm.n_components) + "</td></tr>"
#           print "<tr><td>ObjectName, </td><td>Instance Name,</td><td> GMM Cluster Probability Densities,</td><td>Cluster based on Densities, </td><td>GMM Cluster Probabilities,</td><td> Cluster Number based on probabilities </td></tr>"
          clusterImgs = {}
          for idx in densityPDFs.keys():
              instS = dataInsts[idx].split("/")
#              print "<tr><td>",instS[0],",</td><td>",dataInsts[idx],",</td><td>",
              strIdx = [str(id) for id in densityPDFs[idx]]
#               print " ".join(strIdx),",</td><td>",
#               print str(np.argmax(densityPDFs[idx])),",</td><td>",
              strIdx = [str(id) for id in densityProbs[idx]]
#               print " ".join(strIdx),",</td><td>",
#               print str(np.argmax(densityProbs[idx])),"</td></tr>"
              clusterImgs.setdefault(np.argmax(densityPDFs[idx]), []).append(dataInsts[idx])
#           print "</table>"
#           print "<table><tr><th> Clusters based on Densities</th></tr>"
          
          for key, imgs in clusterImgs.items():
#            print "<tr><td>",str(key),"<td>"
            for img in imgs:
               oN = img.split("/")
#                print '<td><img height="42" width="42" src="https://raw.githubusercontent.com/nispillai/DataSet-GroundedLanguageAcquisition/master/UMBC-RGB-DATASET/' + str(img) + '/' + str(oN[1]) + '_1.png"></td>'
#             print "</tr>"          
#           print "/table>"
           
          exit(0)
          if selectionType == 'max':
            indicesTobeTrained = self.getGMMIndicesfromDenistyPDFs(densityPDFs,densityProbs,learnInstances,instIndices)
          elif selectionType == 'entropy':
            indicesTobeTrained = self.getGMMIndicesfromEntropyPDFs(densityPDFs,densityProbs,learnInstances,instIndices)
          strIndicesToBeTrained = [str(ind) for ind in indicesTobeTrained]
          lineIndices = " ".join(strIndicesToBeTrained)
          os.system("echo \'" + lineIndices + "\' >  GMMIndices/indicesGMMbased.log")
          print (len(indicesTobeTrained),len(list(set(indicesTobeTrained))))
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

      relPoints = totalNeeded
      relDataSets = []
      print (relPoints,len(dataInsts))
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


def setALParameters():
  global gmmPoints,gmmGenerate,iteractionNo,selectionType,dppPts
  gmmPoints = args.gmmPts 
  gmmGenerate = args.generate
  iteractionNo = args.iteration
  selectionType = args.selectionType
  dppPts = args.dppPts
  
  
def getMeaningfulPoints(ds, nDf, tests, configParams) :
#  print(configParams)
  alConfig = configParams['AL']
  inds = configParams['noPts']
  kind = configParams['cat']
  dsPath = configParams['featuresDSPath']
  preFile = configParams['annotationFile']
  alPts = ALPoints(dsPath, preFile)
  global gmmPoints, dppPts, execType, selectionType, iteractionNo,  testInstanceID
  gmmPoints = alConfig['gmmPoints']
  dppPts = alConfig['dppPoints']
  execType = alConfig['execType']
  selectionType = alConfig['selectionType']
  iteractionNo = configParams['iteration']
  testInstanceID = configParams['testNum']
  tIndices = []
  if alConfig['execType'] == 'alpool' or alConfig['execType'] == 'aluncert':
       if alConfig['generatePoints'] == 'yes':
         tIndices = alPts.generateALTrainingInstancesPool(nDf,tests,kind,inds)
       else:
         tIndices = alPts.getALTrainingInstancesPool(nDf,tests,kind,inds)
  elif alConfig['execType']  == 'aldpp' or alConfig['execType']  == 'algmmdpp':

       if alConfig['generatePoints'] == 'yes':
         tIndices = alPts.generateALDPPTrainingInstances(nDf,tests,kind,inds)
       else:
         tIndices = alPts.getALDPPTrainingInstances(nDf,tests,kind,inds)
  elif execType == 'ald2VgmmPool':
#       or execType == 'ald2VgmmUncert':
       if alConfig['generatePoints'] == 'yes':
         tIndices = alPts.generateLangd2VecGMMALTrainingInstances(nDf,tests,kind,inds)
       else:
         tIndices = alPts.getALTrainingInstancesPool(nDf,tests,kind,inds)
  return tIndices 

  
  