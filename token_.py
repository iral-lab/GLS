import numpy as np

# This is the variable that defines how many positive instances a token must
# have before it is deemed useful.
# NOTE: this is different from how many times it appeared for a particular
# instance. For the training that appears to be 1.
MIN_POS_INSTS = 3

class Token:
    """ Class to bundle token (word) related functions and variables """

    def __init__(self, name):
        """
        Initialization function for Token class
        Args: token name ("red")
        Returns: Nothing
        """
        self.name = name
        self.posInstances = np.array([], dtype='object')
        self.negInstances = np.array([], dtype='object')

    def getTokenName(self):
        """
        Function to get the label from class instance
        Args: Token class instance
        Returns: token (label, for ex: "red")
        """
        return self.name

    def extendPositives(self, instName):
        """
        Function to add postive instance (tomato/tomato_1) for this token (red)

        Args: token class instance, positive instance
        Returns: None
        """
        self.posInstances = np.append(self.posInstances, instName)

    def getPositives(self):
        """
        Function to get all postive instances of this token

        Args: token class instance
        Returns: array of positive instances (ex: tomato/tomato_1, ..)
        """
        return self.posInstances

    def extendNegatives(self, *instName):
        """
        Function to add negative instances for this token

        Args: Instance class instance, array of negative instances
        Returns: None
        """
        self.negInstances = np.unique(np.append(self.negInstances, instName))

    def getNegatives(self):
        """
        Function to get all negative instances of this token (ex, "red")

        Args: token class instance
        Returns: array of negative instances (ex: arch/arch_1, ..)
        """
        return self.negInstances

    def clearNegatives(self):
        self.negInstances = np.array([])

    # PAT TODO: check this against repo for indentation
    def getTrainFiles(self, insts, kind):
        """
        This function is to get all training features for this particular token
        >> Find positive instances described for this token
        >> if the token is used less than 3 times, remove it from execution
        >> fetch the feature values from the physical dataset location
        >> find negative instances and fetch the feature values from the physical location
        >> balance the number positive and negative feature samples

        Args: token class instance, complete Instance list, type for learning/testing
        Returns: training features (X) and values (Y)
        """
        instances = insts.to_dict()
        # NOTE: this is not how many times a token was used at all, but how many positive instances it has
        # This means a token count be used 100 times for a particular instance but still not make the cut.
        if len(np.unique(self.posInstances)) < MIN_POS_INSTS:
            return np.array([]), np.array([])
        # print self.name,":",self.posInstances
        features = np.array([])
        negFeatures = np.array([])
        y = np.array([])
        if self.posInstances.shape[0] == 0 or self.negInstances.shape[0] == 0:
            return (features, y)

        if self.posInstances.shape[0] > 0:
            features = np.vstack(instances[inst][0].getFeatures(kind) for inst in self.posInstances)

        if self.negInstances.shape[0] > 0:
            negFeatures = np.vstack(instances[inst][0].getFeatures(kind) for inst in self.negInstances if len(inst) > 1)
        # if length of positive samples are more than the length of negative samples,
        # duplicate negative instances to balance the count
            if len(features) > len(negFeatures):
                c = int(len(features) / len(negFeatures))
                negFeatures = np.tile(negFeatures, (c, 1))

        if self.posInstances.shape[0] > 0 and self.negInstances.shape[0] > 0:
        # if length of positive samples are less than the length of negative samples,
        # duplicate positive samples to balance the count
            if len(negFeatures) > len(features):
                c = int(len(negFeatures) / len(features))
                features = np.tile(features, (c, 1))
        # find trainY for our binary classifier: 1 for positive samples,
        #0 for negative samples
        y = np.concatenate((np.full(len(features), 1), np.full(len(negFeatures), 0)))
        if self.negInstances.shape[0] > 0:
            features = np.vstack([features, negFeatures])
        return(features, y)
