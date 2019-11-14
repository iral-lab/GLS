import cLLML
import samples
from keras.losses import mse, binary_crossentropy
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, BatchNormalization, Activation, Dropout, Conv2D, Conv2DTranspose, Concatenate
from keras.regularizers import l2
from sklearn.preprocessing import MultiLabelBinarizer
from keras.initializers import RandomUniform
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import Model
from keras import metrics
from keras import backend as K
import numpy as np
from keras.layers.merge import concatenate
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


algType = 1
kind = ''

def MLP(inputs, hDims, activationFunction):
   l2_loss = 1e-6
   layer = inputs
   for i in range(len(hDims)):
   	   x = Dense(hDims[i], activation=activationFunction, kernel_regularizer = l2(l2_loss))(layer)
   	   layer = x       	 

   return layer

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
   
    z_mean, z_log_var = args
    epsilon = 1.0
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
   

def VAEM1Model(x_train, hDims, activationFunction):
   image_size = x_train[0].shape[0]
   input_shape = (image_size, )
   inputs = Input(shape=input_shape)
   epochs = 5
   latentDim = hDims[-1]
   print("latent dim ",latentDim)
# VAE model = encoder + decoder
# build encoder model
   enc = inputs
   if len(hDims) > 1:
       enc = MLP(enc, hDims[:-1], activationFunction)
   z_mean = Dense(latentDim, name='z_mean')(enc)
   z_log_var = Dense(latentDim, name='z_log_var')(enc)   
   
# use reparameterization trick to push the sampling out as input
   z = Lambda(sampling, output_shape=(latentDim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
   encoder = Model(inputs, z, name='encoder')
   encoder.summary()

   # build decoder model
   latent_inputs = Input(shape=(latentDim,), name='z_sampling')
   dec = latent_inputs
   if len(hDims) > 1:
       hDims = np.array(hDims[:-1])
       hDims = hDims[::-1]
       print(hDims)
       dec = MLP(latent_inputs, hDims, activationFunction)
   outputs = Dense(image_size, activation=activationFunction)(dec)

# instantiate decoder model
   decoder = Model(latent_inputs, outputs, name='decoder')
   decoder.summary()

# instantiate VAE model
   outputs = decoder(encoder(inputs))
   vae = Model(inputs, outputs, name='vae_mlp')
   reconstruction_loss = binary_crossentropy(inputs,outputs)
   reconstruction_loss *= input_shape
   kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
   kl_loss = K.sum(kl_loss, axis=-1)
   kl_loss *= -0.5
   vae_loss = K.mean(reconstruction_loss + kl_loss)
   vae.add_loss(vae_loss)
   vae.compile(optimizer='adam')
   vae.summary()
        # train the autoencoder
   vae.fit(x_train, epochs=epochs,batch_size=1)
   return encoder,vae

def m2SupervisedLoss(X, Y, y_pred, z2_mean, z2_var, z2, xhat):
   #Calculation for labeled data	  
   # X  -- Samples selected using reparameterized trick 
   # Y - 25 dimension Label Vector. 1 for positive label, 0 for negative label, -1 for unlabeled
   # y_pred - Classifier prediction results from p (y / x)
   # z2 - Latent representation samples - Samples selected using reparameterization trick from z2_var and z2_mean, q_phi (z | y, x)
   # z2_var - z variance from M2 VAE generated from q_phi (z | y, x)
   # z2_mean - z mean from M2 VAE generated from q_phi (z | y, x)
   # xhat - reconstructured X from z and y. p_theta(x|y,z, theta)
   
        
   	    piLog = np.log(2 * np.pi)
  
   	#log(p_z) is gaussian - log prior z
   	    gMz = -0.5 * (piLog + K.square(z2)) # z2 is point selected using reparametric method from z mean and z variance
   	    pLogz = K.sum(gMz, axis=0)
   	    
   	#log(p_y) is binary cross entropy error between classifier output and original y. This is where masking is needed.
   	    pLogy = -1 * binary_crossentropy(Y, y_pred)

   	  	
#    	#log p(x |  y, z) is gaussian    or simple reconstruction error function? -- this is loglikelihood -- included in main function
#    	    gMx = -0.5 * piLog - xhat_var / 2  - (K.square(X  - xhat_mean)) / (2 * K.exp(xhat_var))
#    	    pLogx = K.sum(gMx, axis=1)

   	#log q (z|x, y) is gaussian
   	    gMPz = -0.5 * piLog - z2_var / 2  - (K.square(z2  - z2_mean)) / (2 * K.exp(z2_var))
   	    postLogz = K.sum(gMPz, axis=0)   	  	
   	  		  
#    	    L = pLogz + pLogy + pLogx - postLogz
   	    L = pLogz + pLogy - postLogz
   	    return  L # L is negative objective function. 

def m2UnLabelledLoss(x_samples, y_inputs, qy_classifier, Y_Ubl, Z2_Ubl, Z2_mu_Ubl, Z2_sig_Ubl, PX2_Ubl, z_Samples_Count):
   def loss(x,x_decoded):
   	Lss = 0.0
#    	#mask_val=-1 . unlabeled got -1
#    	mask_val= -1
#    	# Masking -1 values here. Taking only positive and negative tokens
#    	s_mask = K.cast(K.not_equal( y_inputs, mask_val), K.floatx())
#    	y_inputs_L =  y_inputs * s_mask
#    	qy_classifier_L = qy_classifier * s_mask
	   	

#    Y_samples = getUnsupervisedSamples(qy_classifier.shape)

   	for lIjk in range(z_Samples_Count):
   	  Y = Y_Ubl[lIjk]
   	  for uijk in range(z_Samples_Count):
   	     	  ijk = uijk
   	     	  z2_samples = Z2_Ubl[ijk]
   	     	  z2_mu_outs = Z2_mu_Ubl[ijk]
   	     	  z2_sig_outs = Z2_sig_Ubl[ijk]
   	     	  px_outs = PX2_Ubl[ijk]   	
   	     	  
  	 	 
   	     	  nL_lab = m2SupervisedLoss(x_samples,  y_inputs, qy_classifier,  z2_mu_outs, z2_sig_outs, z2_samples, px_outs)
   	     	  nL_lab +=  -1 * binary_crossentropy(x,x_decoded)
   	     	  y_ent = -1 * Y  * K.log(Y)
# #   	     	  from scipy.stats import entropy
# #   	     	  y_ent = entropy(Y)
   	     	  ls = K.sum(K.sum(Y) * -1 * nL_lab + K.sum(y_ent))
#   	     	  ls = K.sum( -1 * nL_lab)

   	     	  Lss += ls

   	return K.sum(Lss)

   return loss  	      	    
   	    
def m2LabelledLoss(x_samples, y_inputs, qy_classifier, Z2, Z2_mu, Z2_sig, PX2, z_Samples_Count):
   def loss(x,x_decoded):
   	Lss = 0.0
   	#mask_val=-1 . unlabeled got -1
   	mask_val= -1
   	# Masking -1 values here. Taking only positive and negative tokens
   	s_mask = K.cast(K.not_equal( y_inputs, mask_val), K.floatx())
   	y_inputs_L =  y_inputs * s_mask
   	qy_classifier_L = qy_classifier * s_mask
   	for ijk in range(z_Samples_Count):
  	 	 l_z2_samples = Z2[ijk]
  	 	 l_z2_mu_outs = Z2_mu[ijk]
  	 	 l_z2_sig_outs = Z2_sig[ijk]
  	 	 l_px_outs = PX2[ijk]   	
  	 	 nL_lab = m2SupervisedLoss(x_samples, y_inputs_L, qy_classifier_L,  l_z2_mu_outs, l_z2_sig_outs, l_z2_samples, l_px_outs)
  	 	 nL_lab +=  -1 * binary_crossentropy(x,x_decoded)
  	 	 ls = K.sum(-1 * nL_lab)
  	 	 Lss += ls

   	return K.sum(Lss)

   return loss  	   	 
   	

   
#def m2loss(px_mu_outs, px_sig_outs, x_samples, y_inputs, z2_samples, z2_mu_outs, z2_sig_outs, qy_latent):
#def m2loss(x_samples, qy_outs, qy_latent, u_z2_mu_outs, u_z2_sig_outs, u_z2_samples, u_px_mu_outs, u_px_sig_outs, y_inputs, l_z2_mu_outs, l_z2_sig_outs, l_z2_samples,  l_px_mu_outs, l_px_sig_outs):
def m2loss(x_samples, y_inputs, qy_classifier,  qy_latent, u_z2_mu_outs, u_z2_sig_outs, u_z2_samples, u_px_outs, l_z2_mu_outs, l_z2_sig_outs, l_z2_samples, l_px_outs):
   def loss(x,x_decoded):
      print(x.shape, x_decoded.shape, x_samples.shape)
      print(y_inputs)
      for i in range(y_inputs.shape[1]):
      	print(y_inputs[i][0])
      	
         	    #mask_val=-1 . unlabeled got -1
      mask_val= -1
   	    # Masking -1 values here. Taking only positive and negative tokens
      s_mask = K.cast(K.not_equal( y_inputs, mask_val), K.floatx())

      y_inputs_L =  y_inputs * s_mask
      qy_classifier_L = qy_classifier * s_mask
      nL_lab = m2SupervisedLoss(x_samples, y_inputs_L, qy_classifier_L,  l_z2_mu_outs, l_z2_sig_outs, l_z2_samples, l_px_outs)

      nL_lab += -1 * binary_crossentropy(x,x_decoded)
#    	    else:
#    	    	nL_lab = K.sum(qy_latent * (L - K.log(qy_latent)), axis=1)
#    	    	   	  	
#   	    return K.sum(nL_lab * -1)

      return K.sum(-1 * nL_lab)
   	    
   return loss  	   

  #m2 encoder
def M2Encoder(xy, hDims, nl_qz, l2_loss, dim_z, lName):
#   qz_layer = Dense(hidden_qz, activation=nl_qz,kernel_regularizer = l2(l2_loss))(xy)
   qz_layer = xy
   if len(hDims) > 0:
         qz_layer = MLP(qz_layer, hDims, nl_qz)
   z2_mu_outs = Dense(dim_z, activation=nl_qz, name=lName + 'm2encodermu',kernel_regularizer = l2(l2_loss))(qz_layer)
   z2_sig_outs = Dense(dim_z, activation=nl_qz, name=lName + 'm2encodersig',kernel_regularizer = l2(l2_loss))(qz_layer)   
   z2_samples = Lambda(sampling, output_shape=(dim_z,), name=lName + 'z2_sample')([z2_mu_outs, z2_sig_outs])
   return   z2_mu_outs, z2_sig_outs, z2_samples

  #m2 decoder   
def M2Decoder( zy_concat, hDims, nl_px, l2_loss, dim_x, lName):
#   px_layer = Dense(hidden_px, activation=nl_px,kernel_regularizer = l2(l2_loss))(zy_concat)
   px_layer = zy_concat
   if len(hDims) > 0:
         px_layer = MLP(px_layer, hDims, nl_px)
   px_outs = Dense(dim_x , activation=nl_px, name=lName + 'm2decoder',kernel_regularizer = l2(l2_loss))(px_layer)
#   px_sig_outs = Dense(dim_x , activation=nl_px, name=lName + 'm2decodersig',kernel_regularizer = l2(l2_loss))(px_layer)    
   return px_outs
  

def VAEM2Model(m1Enc, m1Vae, X, Y, hDims, activationFunction):
   dim_z = hDims[-1]

   dim_y = Y.shape[1]
   epochs = 5
   lrate = 3e-4
   alpha = 0.1
   l2_loss = 1e-6
   p_x = 'gaussian'
   q_z = 'gaussian'
   p_z = 'gaussian'
   p_y = 'uniform'
   nl_px = 'softplus'
   nl_qz = 'softplus'
   nl_qy = 'softplus' 
   #p(x|z,y), q(z|x,y) and q(y|x)

   z_Samples_Count = 10
   
   
   X_m1 = np.array(m1Enc.predict(X))    
   dim_x = X_m1.shape[1]
   #M2  network
   # sampling of data_lab
 
   y_inputs = Input(shape=(dim_y,), name='yv2_input')
   x_samples = Input(shape=(dim_x,), name='xv2_input')
   
   Z2_Ubl = []
   PX2_Ubl = []
   Z2_mu_Ubl = []
   Z2_sig_Ubl = []
   Y_Ubl = []   
   for lIjk in range(z_Samples_Count):
      #classifier network   for unlabeled samples
#      qy_layer = Dense(hidden_qy, activation=nl_qy,kernel_regularizer = l2(l2_loss))(x_samples)
      qy_layer = x_samples
      if len(hDims) > 0:
         qy_layer = MLP(qy_layer, hDims, nl_qy)
      qy_outs = Dense(dim_y, activation=nl_qy,name='m2latent'+ str(lIjk),kernel_regularizer = l2(l2_loss))(qy_layer)
      qy_latent = Dense(dim_y, activation='sigmoid',name='m2latent_predicts'+ str(lIjk),kernel_regularizer = l2(l2_loss))(qy_outs)

      if lIjk == 0:
      	Y_Ubl = qy_latent
      else:
      	Y_Ubl = concatenate([Y_Ubl, qy_latent], axis=0)
   
   
#    yMarginSamples = samples.getS()
#    yMarginSamples = np.array(yMarginSamples)
#    yMS = [K.variable(np.reshape(yMarginSamples[lIjk],(-1,yMarginSamples.shape[1]))) for lIjk in range(yMarginSamples.shape[0]) ]
# 
#    for lIjk in range(yMarginSamples.shape[0]):
#       yS = yMS[lIjk]
#       print(yS)
#       yS = K.reshape(yS, shape=(None,yMarginSamples.shape[1]))
#       print(yS)
#       
      #m2   for unlabeled samples
   hDimSizes = np.array(hDims[:-1])
   hDimsR = hDimSizes[::-1]
   
   for ijk in range(z_Samples_Count):
#            u_xy_concat = concatenate([x_samples,qy_latent])
#            u_xy_concat = concatenate([x_samples,yS])     
            u_xy_concat = concatenate([x_samples,y_inputs])       
            u_z2_mu_outs, u_z2_sig_outs, u_z2_samples = M2Encoder(u_xy_concat, hDimSizes, nl_qz, l2_loss, dim_z,  "unlbl" + str(lIjk) + "-" + str(ijk))
#            u_zy_concat = concatenate([u_z2_samples,qy_latent])
            u_zy_concat = concatenate([u_z2_samples, y_inputs])            
            u_px_outs = M2Decoder( u_zy_concat, hDimsR, nl_px, l2_loss, dim_x,  "unlbl"+ str(lIjk) + "-" + str(ijk))
            if ijk == 0 :
                Z2_Ubl = u_z2_samples
                Z2_mu_Ubl = u_z2_mu_outs
                Z2_sig_Ubl = u_z2_sig_outs
                PX2_Ubl = u_px_outs
            else:
                Z2_Ubl = concatenate([Z2_Ubl, u_z2_samples], axis=0)      
                Z2_mu_Ubl = concatenate([Z2_mu_Ubl, u_z2_mu_outs], axis=0)     
                Z2_sig_Ubl = concatenate([Z2_sig_Ubl, u_z2_sig_outs], axis=0)     
                PX2_Ubl = concatenate([PX2_Ubl, u_px_outs], axis=0)
                

#    # Supervised Model
#    Z2 = []
#    PX2 = []
#    Z2_mu = []
#    Z2_sig = []
#    #m2   for labeled samples
#    for ijk in range(z_Samples_Count):
#       l_xy_concat = concatenate([x_samples,y_inputs])
#       l_z2_mu_outs, l_z2_sig_outs, l_z2_samples = M2Encoder(l_xy_concat, hDimSizes, nl_qz, l2_loss, dim_z,  "lbl" + str(ijk))
#             	
#       l_zy_concat = concatenate([l_z2_samples,y_inputs])
#       l_px_outs = M2Decoder( l_zy_concat, hDimsR, nl_px, l2_loss, dim_x,  "lbl"+ str(ijk))
#       Z = concatenate([l_z2_mu_outs, l_z2_sig_outs, l_z2_samples], axis=0)
#       if ijk == 0:
#       	Z2 = l_z2_samples
#       	Z2_mu = l_z2_mu_outs
#       	Z2_sig = l_z2_sig_outs
#       	PX2 = l_px_outs
#       else:
#       	Z2 = concatenate([Z2, l_z2_samples], axis=0)      
#       	Z2_mu = concatenate([Z2_mu, l_z2_mu_outs], axis=0)     
#       	Z2_sig = concatenate([Z2_sig, l_z2_sig_outs], axis=0)     
#       	PX2 = concatenate([PX2, l_px_outs], axis=0)
# 
# 
 # multi-label classifier y^
   qy_mLabel =  x_samples
   if len(hDims) > 0:
         qy_mLabel = MLP(qy_mLabel, hDims, nl_qy)
#   qy_mLabel = Dense(hidden_qy, activation=nl_qy,kernel_regularizer = l2(l2_loss))(x_samples)
   qy_classifier = Dense(dim_y, activation='sigmoid',name='m2predicts',kernel_regularizer = l2(l2_loss))(qy_mLabel)

# 
#    m2Model = Model(inputs=[x_samples, y_inputs], outputs=[PX2, qy_classifier], name='vaem2model')  ## supervised model with N latent samples
   
   m2Model = Model(inputs=[x_samples, y_inputs], outputs=[PX2_Ubl, qy_classifier], name='vaem2model')  ## unsupervised model with N * N latent samples
 
   m2Model.summary()   
   classModel = Model(inputs=[x_samples], outputs=[qy_classifier])
#   classModel = Model(inputs=[x_samples], outputs=[u_z2_samples])
   optimizer = Adam(lr=lrate)
#    m2Model.compile(optimizer=optimizer, loss=m2LabelledLoss(x_samples, y_inputs, qy_classifier, Z2, Z2_mu, Z2_sig, PX2, z_Samples_Count))   
#   m2Model.compile(optimizer=optimizer, loss=m2UnLabelledLoss(x_samples, yMarginSamples, Y_Ubl, qy_classifier, Z2_Ubl, Z2_mu_Ubl, Z2_sig_Ubl, PX2_Ubl, z_Samples_Count))   
   m2Model.compile(optimizer=optimizer, loss=m2UnLabelledLoss(x_samples, y_inputs, qy_classifier, Y_Ubl, Z2_Ubl, Z2_mu_Ubl, Z2_sig_Ubl, PX2_Ubl, z_Samples_Count))   
#   m2Model.compile(optimizer=optimizer, loss='binary_crossentropy')

#   X2 = np.array([X_m1[i] for i in range(X_m1.shape[1])])
   print("...",X_m1.shape)
   X2 = np.array(X_m1)
   print("...",X2.shape)
 #  history = m2Model.fit([X2,Y], [X2,Y], epochs=epochs,batch_size=1)
   yMarginSamples = samples.getS()
   yMarginSamples1 = np.array(yMarginSamples)
   print( "=====", yMarginSamples1.shape, len(X))
   yMarginSamples = np.array(yMarginSamples1[:len(X),:])
   yMarginSamples1 = []
   for i in range(len(yMarginSamples)):
       yS1 = []
       for ij in range(dim_y - yMarginSamples.shape[1]):
           yS1.extend([0])
       yS1.extend(yMarginSamples[i])
       yMarginSamples1.append(yS1)
   yMarginSamples = np.array(yMarginSamples1)

   print("------>", yMarginSamples.shape, X2.shape, qy_classifier)
   history = m2Model.fit([X2, yMarginSamples], [X2, yMarginSamples], epochs=epochs,batch_size=1)

   return   classModel


def VAEM2Model1(m1Enc, m1Vae, X, Y, hDims, activationFunction):
   dim_z = hDims[-1]

   dim_y = Y.shape[1]
   epochs = 5
   lrate = 3e-4
   alpha = 0.1
   l2_loss = 1e-6
   p_x = 'gaussian'
   q_z = 'gaussian'
   p_z = 'gaussian'
   p_y = 'uniform'
   nl_px = 'softplus'
   nl_qz = 'softplus'
   nl_qy = 'softplus' 
   #p(x|z,y), q(z|x,y) and q(y|x)

   z_Samples_Count = 10
   
   
   X_m1 = np.array(m1Enc.predict(X))    
   dim_x = X_m1.shape[1]
   #M2  network
   # sampling of data_lab
 
   y_inputs = Input(shape=(dim_y,), name='yv2_input')
   x_samples = Input(shape=(dim_x,), name='xv2_input')
   
   Z2_Ubl = []
   PX2_Ubl = []
   Z2_mu_Ubl = []
   Z2_sig_Ubl = []
   Y_Ubl = []   
   for lIjk in range(z_Samples_Count):
      #classifier network   for unlabeled samples
#      qy_layer = Dense(hidden_qy, activation=nl_qy,kernel_regularizer = l2(l2_loss))(x_samples)
      qy_layer = x_samples
      if len(hDims) > 0:
         qy_layer = MLP(qy_layer, hDims, nl_qy)
      qy_outs = Dense(dim_y, activation=nl_qy,name='m2latent'+ str(lIjk),kernel_regularizer = l2(l2_loss))(qy_layer)
      qy_latent = Dense(dim_y, activation='sigmoid',name='m2latent_predicts'+ str(lIjk),kernel_regularizer = l2(l2_loss))(qy_outs)

      if lIjk == 0:
      	Y_Ubl = qy_latent
      else:
      	Y_Ubl = concatenate([Y_Ubl, qy_latent], axis=0)
   
   
#    yMarginSamples = samples.getS()
#    yMarginSamples = np.array(yMarginSamples)
#    yMS = [K.variable(np.reshape(yMarginSamples[lIjk],(-1,yMarginSamples.shape[1]))) for lIjk in range(yMarginSamples.shape[0]) ]
# 
#    for lIjk in range(yMarginSamples.shape[0]):
#       yS = yMS[lIjk]
#       print(yS)
#       yS = K.reshape(yS, shape=(None,yMarginSamples.shape[1]))
#       print(yS)
#       
      #m2   for unlabeled samples
   hDimSizes = np.array(hDims[:-1])
   hDimsR = hDimSizes[::-1]
   
   for ijk in range(z_Samples_Count):
#            u_xy_concat = concatenate([x_samples,qy_latent])
#            u_xy_concat = concatenate([x_samples,yS])     
            u_xy_concat = concatenate([x_samples,y_inputs])       
            u_z2_mu_outs, u_z2_sig_outs, u_z2_samples = M2Encoder(u_xy_concat, hDimSizes, nl_qz, l2_loss, dim_z,  "unlbl" + str(lIjk) + "-" + str(ijk))
#            u_zy_concat = concatenate([u_z2_samples,qy_latent])
            u_zy_concat = concatenate([u_z2_samples, y_inputs])            
            u_px_outs = M2Decoder( u_zy_concat, hDimsR, nl_px, l2_loss, dim_x,  "unlbl"+ str(lIjk) + "-" + str(ijk))
            if ijk == 0 :
                Z2_Ubl = u_z2_samples
                Z2_mu_Ubl = u_z2_mu_outs
                Z2_sig_Ubl = u_z2_sig_outs
                PX2_Ubl = u_px_outs
            else:
                Z2_Ubl = concatenate([Z2_Ubl, u_z2_samples], axis=0)      
                Z2_mu_Ubl = concatenate([Z2_mu_Ubl, u_z2_mu_outs], axis=0)     
                Z2_sig_Ubl = concatenate([Z2_sig_Ubl, u_z2_sig_outs], axis=0)     
                PX2_Ubl = concatenate([PX2_Ubl, u_px_outs], axis=0)
                

#    # Supervised Model
#    Z2 = []
#    PX2 = []
#    Z2_mu = []
#    Z2_sig = []
#    #m2   for labeled samples
#    for ijk in range(z_Samples_Count):
#       l_xy_concat = concatenate([x_samples,y_inputs])
#       l_z2_mu_outs, l_z2_sig_outs, l_z2_samples = M2Encoder(l_xy_concat, hDimSizes, nl_qz, l2_loss, dim_z,  "lbl" + str(ijk))
#             	
#       l_zy_concat = concatenate([l_z2_samples,y_inputs])
#       l_px_outs = M2Decoder( l_zy_concat, hDimsR, nl_px, l2_loss, dim_x,  "lbl"+ str(ijk))
#       Z = concatenate([l_z2_mu_outs, l_z2_sig_outs, l_z2_samples], axis=0)
#       if ijk == 0:
#       	Z2 = l_z2_samples
#       	Z2_mu = l_z2_mu_outs
#       	Z2_sig = l_z2_sig_outs
#       	PX2 = l_px_outs
#       else:
#       	Z2 = concatenate([Z2, l_z2_samples], axis=0)      
#       	Z2_mu = concatenate([Z2_mu, l_z2_mu_outs], axis=0)     
#       	Z2_sig = concatenate([Z2_sig, l_z2_sig_outs], axis=0)     
#       	PX2 = concatenate([PX2, l_px_outs], axis=0)
# 
# 
 # multi-label classifier y^
   qy_mLabel =  x_samples
   if len(hDims) > 0:
         qy_mLabel = MLP(qy_mLabel, hDims, nl_qy)
#   qy_mLabel = Dense(hidden_qy, activation=nl_qy,kernel_regularizer = l2(l2_loss))(x_samples)
   qy_classifier = Dense(dim_y, activation='sigmoid',name='m2predicts',kernel_regularizer = l2(l2_loss))(qy_mLabel)

# 
#    m2Model = Model(inputs=[x_samples, y_inputs], outputs=[PX2, qy_classifier], name='vaem2model')  ## supervised model with N latent samples
   
   m2Model = Model(inputs=[x_samples, y_inputs], outputs=[PX2_Ubl, qy_classifier], name='vaem2model')  ## unsupervised model with N * N latent samples
 
   m2Model.summary()   
   classModel = Model(inputs=[x_samples], outputs=[qy_classifier])
#   classModel = Model(inputs=[x_samples], outputs=[u_z2_samples])
   optimizer = Adam(lr=lrate)
#    m2Model.compile(optimizer=optimizer, loss=m2LabelledLoss(x_samples, y_inputs, qy_classifier, Z2, Z2_mu, Z2_sig, PX2, z_Samples_Count))   
#   m2Model.compile(optimizer=optimizer, loss=m2UnLabelledLoss(x_samples, yMarginSamples, Y_Ubl, qy_classifier, Z2_Ubl, Z2_mu_Ubl, Z2_sig_Ubl, PX2_Ubl, z_Samples_Count))   
   m2Model.compile(optimizer=optimizer, loss=m2UnLabelledLoss(x_samples, y_inputs, qy_classifier, Y_Ubl, Z2_Ubl, Z2_mu_Ubl, Z2_sig_Ubl, PX2_Ubl, z_Samples_Count))   
#   m2Model.compile(optimizer=optimizer, loss='binary_crossentropy')

#   X2 = np.array([X_m1[i] for i in range(X_m1.shape[1])])
   print("...",X_m1.shape)
   X2 = np.array(X_m1)
   print("...",X2.shape)
 #  history = m2Model.fit([X2,Y], [X2,Y], epochs=epochs,batch_size=1)
   yMarginSamples = samples.getS()
   yMarginSamples1 = np.array(yMarginSamples)
   print( "=====", yMarginSamples1.shape, len(X))
   yMarginSamples = np.array(yMarginSamples1[:len(X),:])
   yMarginSamples1 = []
   for i in range(len(yMarginSamples)):
       yS1 = []
       for ij in range(dim_y - yMarginSamples.shape[1]):
           yS1.extend([0])
       yS1.extend(yMarginSamples[i])
       yMarginSamples1.append(yS1)
   yMarginSamples = np.array(yMarginSamples1)

   print("------>", yMarginSamples.shape, X2.shape, qy_classifier)
   history = m2Model.fit([X2, yMarginSamples], [X2, yMarginSamples], epochs=epochs,batch_size=1)

   return   classModel

def VAEM2Test(m1Enc, m1Vae, m2Vae, testX, testY):

   X_m1 = np.array(m1Enc.predict(testX))    
   X2 = np.array(X_m1)
   ynew = m2Vae.predict(X2)
   proba = np.transpose(ynew)
   return proba
#    # show the inputs and predicted outputs
#    for i in range(len(X2)):
#        print("X %s, Predicted=%s, class %s" % (i, ynew[i], np.argmax(ynew[i])))
#    accs = []
#    precs = []
#    recs = []
#    f1ss = []
   yH1 = [0  if j < 0.5 else 1 for i in range(len(X2)) for j in ynew[i] ]
   yTrue = [j for i in range(len(X2)) for j in testY[i]]
#    for i in range(len(X2)):
#    	yH = [0  if j < 0.5 else 1 for j in ynew[i] ]
#    	(prec,rec,f1s,sp) = precision_recall_fscore_support(testY[i], yH, average='macro')
#    	acc = accuracy_score(testY[i], yH)
#    	accs.append(float(acc))
#    	precs.append(prec)
#    	recs.append(rec)
#    	f1ss.append(f1s)
#    	print(acc,prec,rec,f1s)
#    print(accs)
#    print(float(sum(accs))/float(len(accs)), float(sum(precs))/float(len(precs)), float(sum(recs))/float(len(recs)), float(sum(f1ss))/float(len(f1ss)) )   
   acc = accuracy_score(yTrue, yH1)
   (prec,rec,f1s,sp) = precision_recall_fscore_support(yTrue, yH1, average='macro')
#   print("SupervisedTotal --LD1: %s, IDM1: %s, LD2: %s, IDM2: %s, Accuracy: %s, Precision: %s, Recall: %s, F1-Score: %s" % (ld,iDim,ld2,iDim2,acc,prec,rec,f1s))
   print("Macro Accuracy: ", acc, " Precision: ", prec, " Recall: ", rec, " F1-Score: ", f1s)
   (prec,rec,f1s,sp) = precision_recall_fscore_support(yTrue, yH1, average='micro')
#   print("SupervisedTotal --LD1: %s, IDM1: %s, LD2: %s, IDM2: %s, Accuracy: %s, Precision: %s, Recall: %s, F1-Score: %s" % (ld,iDim,ld2,iDim2,acc,prec,rec,f1s))
   print("Micro Accuracy: ", acc, " Precision: ", prec, " Recall: ", rec, " F1-Score: ", f1s)   
     

def getVAEFeatures(configParams, insts, tkns, tests, tobeTestedTokens, allInstTokens):
   vaeConf = configParams['VAE']
   h1Dims = vaeConf['v1HiddenDims']
   h1activationFunction = vaeConf['v1activationFn']
   h2Dims = vaeConf['v2HiddenDims']
   h2activationFunction = vaeConf['v2activationFn']   
   global algType, kind
   kind = configParams['cat']
   algType = vaeConf['testOption']
   X_features = []
   vaeFeatures = []
   prob = []
   if algType == 1: 
   #M1 with One VAE,
       X_features = np.vstack(insts[inst][0].getFeatures(kind) for inst in insts.keys() if inst not in tests)
       encoder,vae = VAEM1Model(X_features, h1Dims, h1activationFunction)
       X_features = np.vstack([insts[inst][0].getFeatures(kind) for inst in insts.keys()])
       vaeFeatures = encoder.predict(X_features)
   elif algType == 2: 
   #M1 with Grouped VAE
       (xImages_c, trainFeatures_c, labels_c) = cLLML.getImagesAndFeatures(insts, tests, tobeTestedTokens, 'rgb', allTknInsts=allInstTokens)
       (xImages_s, trainFeatures_s, labels_s) = cLLML.getImagesAndFeatures(insts, tests, tobeTestedTokens, 'shape', allTknInsts=allInstTokens)
       trainFeatures_c = np.array(trainFeatures_c)
       trainFeatures_s = np.array(trainFeatures_s)
       encoder_c,vae_c = VAEM1Model(trainFeatures_c, h1Dims, h1activationFunction)
       encoder_s,vae_s = VAEM1Model(trainFeatures_s, h1Dims, h1activationFunction)
       (xImages_c, trainFeatures_c, labels_c) = cLLML.getImagesAndFeatures(insts, tests, tobeTestedTokens, 'rgb', True, allTknInsts=allInstTokens)
       (xImages_s, trainFeatures_s, labels_s) = cLLML.getImagesAndFeatures(insts, tests, tobeTestedTokens, 'shape', True, allTknInsts=allInstTokens)
       (xImages_o, trainFeatures_o, labels_o) = cLLML.getImagesAndFeatures(insts, tests, tobeTestedTokens, 'object', True, allTknInsts=allInstTokens)	
       trainFeatures_c = np.array(trainFeatures_c)
       trainFeatures_s = np.array(trainFeatures_s)
       trainFeatures_c = encoder_c.predict(trainFeatures_c)
       trainFeatures_s = encoder_s.predict(trainFeatures_s)
       X_features = trainFeatures_o
       vaeFeatures = []
       for ind in range(len(trainFeatures_o)):
          nFeature = list(trainFeatures_c[ind])
          nFeature.extend(trainFeatures_s[ind])
          vaeFeatures.append(nFeature)

   elif algType == 3:
   #  M2 with One VAE
    if configParams['inputFeatures'] == 'kernelDescriptors'  :
       (xImages, trainFeatures, labels) = cLLML.getImagesAndFeatures(insts, tests, tobeTestedTokens, kind, labelsForImages=False, allTknInsts=allInstTokens)
       (posLabels, negLabels, unLabels, posNegUnLabelBinaryVector) = cLLML.getPosNegUnlabeledTokens(insts, tests, tobeTestedTokens, kind, labelsForImages=False, allTknInsts=allInstTokens)
       trainFeatures = np.array(trainFeatures)
       encoder,vae = VAEM1Model(trainFeatures, h1Dims, h1activationFunction)
       m2Vae = VAEM2Model(encoder, vae, trainFeatures, posNegUnLabelBinaryVector, h2Dims, h2activationFunction)
#        (xTestImages, testFeatures, testlabels) = cLLML.getImagesAndFeatures(insts, tests, tobeTestedTokens, kind, labelsForImages=False, onlyTestData=True, allTknInsts=allInstTokens)
#        (testPosLabels, testNegLabels, testUnLabels, testPosNegUnLabelBinaryVector) = cLLML.getPosNegUnlabeledTokens(insts, tests, tobeTestedTokens, kind, labelsForImages=False, onlyTestData=True, allTknInsts=allInstTokens)
       (xTestImages, testFeatures, testlabels) = cLLML.getImagesAndFeatures(insts, tests, tobeTestedTokens, kind, labelsForImages=False, includetestData=True, allTknInsts=allInstTokens)
       (testPosLabels, testNegLabels, testUnLabels, testPosNegUnLabelBinaryVector) = cLLML.getPosNegUnlabeledTokens(insts, tests, tobeTestedTokens, kind, labelsForImages=False, includetestData=True, allTknInsts=allInstTokens)
       testFeatures1 = np.array(testFeatures)		
       X_features = testFeatures
       prob = VAEM2Test(encoder, vae, m2Vae, testFeatures1,  testPosNegUnLabelBinaryVector)
       
      
   return list(X_features), list(vaeFeatures), prob