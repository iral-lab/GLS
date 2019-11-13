from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16preprocess
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19preprocess
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inceptionv3preprocess
import keras
import keras_applications
import numpy as np

class preTrained:
 def __init__(self):
  self.name = "PreTrainedNNs"

 def pretrainedNN(self,img_path,nn="resnet50"):
   img = image.load_img(img_path, target_size=(224, 224))
   if nn == "inceptionv3" or nn == "inceptionResnetV2":
     img = image.load_img(img_path, target_size=(299, 299))
   elif nn == "nasnetlarge":
     img = image.load_img(img_path, target_size=(331, 331))

   x = image.img_to_array(img)
   x = np.expand_dims(x, axis=0)
   model = ()

   if nn == "resnet50":
      model,x = self.resNet50Pretrained(x)
   elif nn == "vgg16":
      model,x = self.vgg16Pretrained(x)
   elif nn == "vgg19":
      model,x = self.vgg19Pretrained(x)
   elif nn == "inceptionv3":
      model,x = self.inceptionv3Pretrained(x)
   elif nn == "resnext101":
      model,x = self.resnext101Pretrained(x)
   elif nn == "inceptionResnetV2":
      model,x = self.inceptionResNetV2Pretrained(x)
   elif nn == "nasnetlarge":
      model,x = self.nasnetLargePretrained(x)

   preds = model.predict(x)
#   model.summary()
   predAr = decode_predictions(preds, top=3)[0]
   pred = nn + " ==> "
   pred = ""
   for i in range(len(predAr)):
     pred += ":".join([str(item) for item in predAr[i][1:]]) + "-"
   return pred

 def resNet50Pretrained(self,x):
   model = ResNet50(weights='imagenet')
   x = preprocess_input(x)
   return model,x

 def vgg16Pretrained(self,x):
#   model = VGG16(weights='imagenet', include_top=False)
   model = VGG16(weights='imagenet')
   x = vgg16preprocess(x)
   return model,x
 
 def vgg19Pretrained(self,x):
#   model = VGG19(weights='imagenet', include_top=False)
   model = VGG19(weights='imagenet')
   x = vgg19preprocess(x)
   return model,x

 def inceptionv3Pretrained(self,x):
#   model = InceptionV3(weights='imagenet', include_top=False)
   model = InceptionV3(weights='imagenet')
   x = inceptionv3preprocess(x)
   return model,x

 def inceptionResNetV2Pretrained(self,x):
   model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True, weights='imagenet')
   x = inceptionv3preprocess(x)
   return model,x

 def resnext101Pretrained(self,x):
   model = keras_applications.resnext.ResNeXt101(weights='imagenet',backend=keras.backend,layers=keras.layers,models=keras.models,utils=keras.utils)
   x = preprocess_input(x)
   return model,x

 def nasnetLargePretrained(self,x):
   model = keras.applications.nasnet.NASNetLarge(include_top=True, weights='imagenet')
   x = preprocess_input(x)
   return model,x
