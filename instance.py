import numpy as np
from pandas import read_table

from category import Category

# PAT TODO: E:42 Undefined variable dsPath (from 'globals')
dPath = "../"
dsPath = dPath + ARGS.visfeat

class Instance(Category):
    """ Class to bundle instance wise functions and variables """

    def __init__(self, name, num):
        """
        Initialization function for Instance class
        Args: instance name, category number of this instance
        Returns: Nothing
        """
        super().__init__(self)
        self.name = name
        self.catNum = num
        self.gT = {}
        self.tokens = np.array([])
        self.negs = None

    def getName(self):
        """
        Function to get the instance name
        Args: Instance class instance
        Returns: instance name
        """
        return self.name

    def getFeatures(self, kind):
        """
        Function to find the complete dataset file path (.../arch/arch_1/arch_1_rgb.log)
        where the visual feaures are stored, read the features from the file, and return

        Args: Instance class instance, type of features(rgb, shape, or object)
        Returns: feature set
        """
        instName = self.name
        instName.strip()
        ar1 = instName.split("/")
        path1 = "/".join([dsPath, instName])
        path = path1 + "/" + ar1[1] + "_" + kind + ".log"
        featureSet = read_table(path, sep=',', header=None)
        return featureSet.values

    def addNegatives(self, negs):
        """
        Function to add negative instances

        Args: Instance class instance, array of negative instances
        Returns: None
        """
        add = lambda x: np.unique(map(str.strip, x))
        self.negs = add(negs)

    def getNegatives(self):
        """
        Function to get the list of negative instances

        Args: Instance class instance
        Returns: array of negative instances
        """
        return self.negs

    def addTokens(self, tkn):
        """
        Function to add a word (token) describing this instance to the array of tokens
        Args: Instance class instance, word
        Returns: None
        """
        self.tokens = np.append(self.tokens, tkn)

    def getTokens(self):
        """
        Function to get array of tokens which humans used to describe this instance
        Args: Instance class instance
        Returns: array of words (tokens)
        """
        return self.tokens
