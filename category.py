import random

import numpy as np

class Category:
    """ Class to bundle our dataset functions and variables category wise. """

    def __init__(self, name):
        """
        Initialization function for category class
        Args: category name
        Returns: Nothing
        """
        self.catNums = np.array([], dtype='object')
        self.name = name

    def getName(self):
        """
        Function to get the category name
        Args: category class instance
        Returns: category name
        """
        return self.name

    def addCategoryInstances(self, *num):
        """
        Function to add a new instance number to the category
        Args: category class instance
        Returns: None
        """
        self.catNums = np.unique(np.append(self.catNums, num))

    def chooseOneInstance(self):
        """
        Function to select one random instance from this category for testing
        Args: category class instance
        Returns: Randomly selected instance name
        """
        r = random.randint(0, self.catNums.size - 1)
        instName = self.name + "/" + self.name + "_" + self.catNums[r]
        return instName
