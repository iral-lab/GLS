# USAGE

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from CNN.Models.smallervggnet import SmallerVGGNet
from keras.layers import Dense, GlobalAveragePooling2D
import matplotlib.pyplot as plt
from keras.layers import Dense
from imutils import paths
import numpy as np
import argparse
from keras.models import Model
import random
import pickle
import keras
import cv2
import os
import keras_applications
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16preprocess
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19preprocess
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inceptionv3preprocess

class MultiLabel:
   def __init__(self,name="ML"):
      self.name = name


   def getModel(self,netName,IMAGE_DIMS,classes_):
     model = ()
     modelExtract  = ()
     if netName == "smallervggnet":
         model = SmallerVGGNet.build(
        width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
        depth=IMAGE_DIMS[2], classes=len(classes_),
        finalAct="sigmoid")
         modelExtract = Model(inputs=model.inputs, outputs=model.layers[-3].output)
     else:
         base_model = model
         x = model
         if netName == "nasnetlarge":
           base_model = keras.applications.nasnet.NASNetLarge(weights='imagenet', include_top=False)
           x = base_model.output
           x = GlobalAveragePooling2D()(x)    
         elif netName == "resnet50":
           base_model = ResNet50(weights='imagenet', include_top=False)
           x = base_model.output
           x = GlobalAveragePooling2D()(x)              
         elif netName == "vgg16":
             base_model = VGG16(weights='imagenet', include_top=False)
             x = base_model.output
             x = GlobalAveragePooling2D()(x)                
         elif netName == "vgg19":
             base_model = VGG19(weights='imagenet', include_top=False)      
             x = base_model.output
             x = GlobalAveragePooling2D()(x)                 
         elif netName == "inceptionv3":
             base_model = InceptionV3(weights='imagenet', include_top=False)
             x = base_model.output
             x = GlobalAveragePooling2D()(x)                 
         elif netName == "resnext101":
             base_model = keras_applications.resnext.ResNeXt101(weights='imagenet', backend=keras.backend,layers=keras.layers,models=keras.models,utils=keras.utils) 
             x = base_model.output
         elif netName == "inceptionResnetV2":
             base_model =  keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet')
             x = base_model.output
             x = GlobalAveragePooling2D()(x)      
         # let's add a fully-connected layer
         x = Dense(1024, activation='relu')(x)
         predictions = Dense(len(classes_),activation='sigmoid')(x)
         model = Model(inputs=base_model.input, outputs=predictions)
         for i,layer in enumerate(base_model.layers):
             layer.trainable = False
#             print(i, layer.name)
             
#         model.summary()
         modelExtract = Model(inputs=model.inputs, outputs=model.layers[-2].output)
         
     modelExtract.summary()

     model.summary()

     return model, modelExtract
     
   def fineTuneModel(self,model,trainX, trainY,netName,BS,EPOCHS, verbose=1) :
#        for i, layer in enumerate(model.layers):
#            print(i, layer.name)
       layerNos = { "nasnetlarge" : 1038, "resnet50": 175, "vgg16" : 19, "vgg19": 22, "inceptionv3" : 311, "resnext101": 477, "inceptionResnetV2":780}
       layerNo = layerNos[netName]            
                                   
       for layer in model.layers[:layerNo]:
           layer.trainable = False
       for layer in model.layers[layerNo:]:
           layer.trainable = True
           
       from keras.optimizers import SGD
       model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy')
       model.fit(trainX, trainY,batch_size = BS,epochs=EPOCHS, verbose=1)
       return model

   def preProcessImages(self,imagePaths,netName):
    IMAGE_DIMS = (224, 224, 3)
    if netName == "smallervggnet":
        IMAGE_DIMS = (96, 96, 3)
    elif netName == "nasnetlarge":
        IMAGE_DIMS = (331, 331,3)
    elif netName ==  "inceptionv3" or netName == "inceptionResnetV2":
        IMAGE_DIMS = (299, 299,3)
              
   # loop over the input images
   # initialize the data and labels
    data = []
    for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)
        data.append(image)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0  
    return  data, IMAGE_DIMS     
               

   def classify(self,imagePaths,labels,modelPath,lbin,plot, xImagesAll,netName="smallervggnet",cNNTune=0):
       
    # initialize the number of epochs to train for, initial learning rate,
    # batch size, and image dimensions
    EPOCHS = 1
    INIT_LR = 1e-3
    BS = 1
    
    # grab the image paths and randomly shuffle them
    print("[INFO] loading images...")

    data, IMAGE_DIMS = self.preProcessImages(imagePaths,netName)
    labels = np.array(labels)
    print("[INFO] data matrix: {} images ({:.2f}MB)".format(
	len(imagePaths), data.nbytes / (1024 * 1000.0)))

    # binarize the labels using scikit-learn's special multi-label
    # binarizer implementation
    print("[INFO] class labels:")
    mlb = MultiLabelBinarizer()
    labels = mlb.fit_transform(labels)

    # loop over each of the possible class labels and show them
    for (i, label) in enumerate(mlb.classes_):
       print("{}. {}".format(i + 1, label))
    # partition the data into training and testing splits using 80% of
    # the data for training and the remaining 20% for testing
#    (trainX, testX, trainY, testY) = train_test_split(data,
#	labels, test_size=0.2, random_state=42)
    trainX = data
    trainY = labels
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

    # initialize the model using a sigmoid activation as the final layer
    # in the network so we can perform multi-label classification
    print("[INFO] compiling model...")
    model, modelExtract = self.getModel(netName,IMAGE_DIMS,mlb.classes_)
    # initialize the optimizer (SGD is sufficient)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

    # compile the model using binary cross-entropy rather than
    # categorical cross-entropy -- this may seem counterintuitive for
    # multi-label classification, but keep in mind that the goal here
    # is to treat each output label as an independent Bernoulli
    # distribution
    if cNNTune == 'yes':
       model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

        # train the network
       print("[INFO] training network...")
       H = model.fit(trainX, trainY,batch_size = BS,epochs=EPOCHS, verbose=1)
    if netName != "smallervggnet":
      if cNNTune == 'yes':
        model = self.fineTuneModel(model,trainX, trainY,netName,BS,EPOCHS, 1)
    # save the model to disk
#     print("[INFO] serializing network...")
#     model.save(modelPath)

    # save the multi-label binarizer to disk
#     print("[INFO] serializing label binarizer...")
#     f = open(lbin, "wb")
#     f.write(pickle.dumps(mlb))
#     f.close()
    if cNNTune == 'yes':
    # plot the training loss and accuracy
     plt.style.use("ggplot")
     plt.figure()
     N = EPOCHS
    
     plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
#    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
     plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
#    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
     plt.title("Training Loss and Accuracy")
     plt.xlabel("Epoch #")
     plt.ylabel("Loss/Accuracy")
     plt.legend(loc="upper left")
     plt.savefig(plot)
    dataAll, IMAGE_DIMS = self.preProcessImages(xImagesAll,netName)
    featuresAll = modelExtract.predict(dataAll)
    print(featuresAll.shape)
    
    return model, modelExtract, mlb,featuresAll


   def test(self,imagePaths,model,mlb,netName="smallervggnet"):
       data, IMAGE_DIMS = self.preProcessImages(imagePaths,netName)
       print(len(data))

       # classify the input image then find the indexes of the two class
       print("[INFO] classifying image...")
       proba = model.predict(data)
       proba = np.transpose(proba)
       return proba
       
