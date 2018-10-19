#!/usr/bin/env python
# coding: utf-8

# In[8]:


from six.moves import xrange
import os
import random
import datetime
import re
import math
import logging
import numpy as np
from math import ceil
import skimage.transform
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import keras.activations as Activation
from keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt
import multiprocessing
from keras import regularizers
import rawpy
import glob


# In[9]:


#Global Parameters
PATCH_SIZE = 512


# In[17]:


data_dir = './dataset/Sony/'
checkpoint_dir = './result_Sony/'
result_dir = './result_Sony/'

# In[6]:


def unet():
        "unet for image reconstruction"
        model_input = KL.Input(shape=(None,None,4),name="u_net_input")
        conv1  = KL.Conv2D(48, (3, 3), strides=(1, 1), name='conv1', use_bias=True,padding="same")(model_input)
        conv1  = KL.LeakyReLU(alpha=0.2)(conv1)
        conv1a = KL.Conv2D(48, (3, 3), strides=(1, 1), name='conv1a', use_bias=True,padding="same")(conv1)
        conv1a = KL.LeakyReLU(alpha=0.2)(conv1a)
        P1     = KL.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same",name="max_pooling_1")(conv1a)

        conv2  = KL.Conv2D(96, (3, 3), strides=(1, 1), name='conv2', use_bias=True,padding="same")(P1)
        conv2  = KL.LeakyReLU(alpha=0.2)(conv2)
        conv2a = KL.Conv2D(96, (3, 3), strides=(1, 1), name='conv2a', use_bias=True,padding="same")(conv2)
        conv2a = KL.LeakyReLU(alpha=0.2)(conv2a)
        P2     = KL.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same",name="max_pooling_2")(conv2a)

        conv3  = KL.Conv2D(192, (3, 3), strides=(1, 1), name='conv3', use_bias=True,padding="same")(P2)
        conv3  = KL.LeakyReLU(alpha=0.2)(conv3)
        conv3a = KL.Conv2D(192, (3, 3), strides=(1, 1), name='conv3a', use_bias=True,padding="same")(conv3)
        conv3a = KL.LeakyReLU(alpha=0.2)(conv3a)
        P3     = KL.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same",name="max_pooling_3")(conv3a)

        conv4  = KL.Conv2D(384, (3, 3), strides=(1, 1), name='conv4', use_bias=True,padding="same")(P3)
        conv4  = KL.LeakyReLU(alpha=0.2)(conv4)
        conv4a = KL.Conv2D(384, (3, 3), strides=(1, 1), name='conv4a', use_bias=True,padding="same")(conv4)
        conv4a = KL.LeakyReLU(alpha=0.2)(conv4a)
        P4     = KL.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same",name="max_pooling_4")(conv4a)

        conv5  = KL.Conv2D(768, (3, 3), strides=(1, 1), name='conv5', use_bias=True,padding="same")(P4)
        conv5  = KL.LeakyReLU(alpha=0.2)(conv5)
        conv5a = KL.Conv2D(768, (3, 3), strides=(1, 1), name='conv5a', use_bias=True,padding="same")(conv5)
        conv5a = KL.LeakyReLU(alpha=0.2)(conv5a)
        P5     = KL.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same",name="max_pooling_5")(conv5a)

        
        up4      = KL.Deconvolution2D(nb_filter=384, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv4',border_mode='same')(P5)
        up4      = KL.LeakyReLU(alpha=0.2)(up4)
        C4       = KL.Concatenate()([P4,up4])
        conv4_u  = KL.Conv2D(384, (3, 3), strides=(1, 1), name='conv4_u', use_bias=True,padding="same")(C4)
        conv4_u  = KL.LeakyReLU(alpha=0.2)(conv4_u)
        conv4a_u = KL.Conv2D(384, (3, 3), strides=(1, 1), name='conv4a_u', use_bias=True,padding="same")(conv4_u) 
        conv4a_u = KL.LeakyReLU(alpha=0.2)(conv4a_u)

        up3      = KL.Deconvolution2D(nb_filter=192, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv3',border_mode='same')(conv4a_u)
        up3      = KL.LeakyReLU(alpha=0.2)(up3)
        C3       = KL.Concatenate()([P3,up3])
        conv3_u  = KL.Conv2D(192, (3, 3), strides=(1, 1), name='conv3_u', use_bias=True,padding="same")(C3)
        conv3_u  = KL.LeakyReLU(alpha=0.2)(conv3_u)
        conv3a_u = KL.Conv2D(192, (3, 3), strides=(1, 1), name='conv3a_u', use_bias=True,padding="same")(conv3_u)
        conv3a_u = KL.LeakyReLU(alpha=0.2)(conv3a_u)
        
 
        up2      = KL.Deconvolution2D(nb_filter=96, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv2',border_mode='same')(conv3a_u)
        up2      = KL.LeakyReLU(alpha=0.2)(up2)
        C2       = KL.Concatenate()([P2,up2])
        conv2_u  = KL.Conv2D(96, (3, 3), strides=(1, 1), name='conv2_u', use_bias=True,padding="same")(C2)
        conv2_u  = KL.LeakyReLU(alpha=0.2)(conv2_u)
        conv2a_u = KL.Conv2D(96, (3, 3), strides=(1, 1), name='conv2a_u', use_bias=True,padding="same")(conv2_u)
        conv2a_u = KL.LeakyReLU(alpha=0.2)(conv2a_u)
        
        up1      = KL.Deconvolution2D(nb_filter=48, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv1',border_mode='same')(conv2a_u)
        up1      = KL.LeakyReLU(alpha=0.2)(up1)
        C1       = KL.Concatenate()([P1,up1])
        conv1_u  = KL.Conv2D(48, (3, 3), strides=(1, 1), name='conv1_u', use_bias=True,padding="same")(C1)
        conv1_u  = KL.LeakyReLU(alpha=0.2)(conv1_u)
        conv1a_u = KL.Conv2D(48, (3, 3), strides=(1, 1), name='conv1a_u', use_bias=True,padding="same")(conv1_u) 
        conv1a_u = KL.LeakyReLU(alpha=0.2)(conv1a_u)

        up0      = KL.Deconvolution2D(nb_filter=24, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv0',border_mode='same')(conv1a_u)
        up0      = KL.LeakyReLU(alpha=0.2)(up0)
        C0       = KL.Concatenate()([model_input,up0])
        conv0_u  = KL.Conv2D(24, (3, 3), strides=(1, 1), name='conv0_u', use_bias=True,padding="same")(C0)
        conv0_u  = KL.LeakyReLU(alpha=0.2)(conv0_u)
        conv0a_u = KL.Conv2D(24, (3, 3), strides=(1, 1), name='conv0a_u', use_bias=True,padding="same")(conv0_u)
        conv0a_u = KL.LeakyReLU(alpha=0.2)(conv0a_u)

        x = KL.Conv2D(12,(1,1),strides=(1,1),name='convr',use_bias=True,padding="same")(conv0a_u)
        x = KL.LeakyReLU(alpha=0.2)(x)
        model_output = KL.Lambda(lambda t:tf.depth_to_space(t,2))(x)

        model = KM.Model(inputs=model_input,outputs=model_output)
        model.summary()
        return model 


# In[7]:


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

# In[ ]:


class Dataset(object):

        def __init__(self,data_mode):
                if data_mode=='train':
                        data_list = './train_list.txt'
                elif data_mode=='dev':
                        data_list = './val_list.txt'
                elif data_mode=='test':
                        data_list = './test_list.txt'
                else:
                        print("wront parameters for data_mode")
                        exit()
                with open(data_list) as f:
                        self.image_pairs = [line.strip().split(' ')[:2] for line in f.readlines()]
        
        def load_image(self,image_pair):
                gt_path    = image_pair[1]
                gt_fn      = os.path.basename(gt_path)
                in_path    = image_pair[0]
                in_fn      = os.path.basename(in_path)
                
                in_expo    = float(in_fn[9:-5])
                gt_expo    = float(gt_fn[9:-5])
                ratio      = min(gt_expo/in_expo,300)
                
                noise_img = pack_raw(rawpy.imread(in_path))*ratio              
                gt_img    = np.float32((rawpy.imread(gt_path).postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16))/65535.0)
                
                #crop in Patch Size
                H = noise_img.shape[0]
                W = noise_img.shape[1]
                
                xx = np.random.randint(0,W-PATCH_SIZE)
                yy = np.random.randint(0,H-PATCH_SIZE)
                input_patch = noise_img[yy:yy+PATCH_SIZE,xx:xx+PATCH_SIZE,:]
                gt_patch    = gt_img[yy*2:yy*2+PATCH_SIZE*2,xx*2:xx*2+PATCH_SIZE*2,:]
                
                if np.random.randint(2,size=1)[0] == 1: # random flip
                    input_patch = np.flip(input_patch,axis = 0)
                    gt_patch    = np.flip(gt_patch,axis = 0)
                if np.random.randint(2,size=1)[0] ==1: #random mirror
                    input_patch = np.flip(input_patch,axis = 1)
                    gt_patch    = np.flip(gt_patch,axis = 1)
                if np.random.randint(2,size=1)[0] ==1: #random transpose
                    input_patch = np.transpose(input_patch,(1,0,2))
                    gt_patch    = np.transpose(gt_patch,(1,0,2))
                input_patch = np.mimimum(input_patch,1.0)
                return input_patch,gt_patch
            
            
def data_generator(dataset,batch_size):
        image_pairs   = np.copy(dataset.image_pairs)
        while True:
                try:
                        image_pair   = random.sample(image_pair,batch_size)
                        batch_noise_image  = np.zeros((batch_size,PATCH_SIZE,PATCH_SIZE,4)
                        batch_target_image = np.zeros((batch_size,PATCH_SIZE*2,PATCH_SIZE*2,3))
                        for i in range(batch_size):
                            noise_image,target_img = dataset.load_image(image_pair[i])
                            batch_noise_image[i,:,:,:]   = noise_image
                            batch_target_image[i,:,:,:]  = target_image
                        inputs = [batch_noise_image]
                        outputs = [batch_target_image]
                        yield inputs,outputs
                except (GeneratorExit, KeyboardInterrupt):
                        raise      

class DenoiseUnet():
        """This class encapsulates the Unet model functionality."""

        def __init__(self,mode,weights,model_dir):
                """
                mode: Either "training" or "inference"
                model_dir: Directory to save training logs and trained weights
                """
                self.model_dir = model_dir
                if mode=="train" and weights=="new":
                        self.set_log_dir()
                self.keras_model = unet()

        def find_last(self):
                """Finds the latest weights file
                Returns:
                        log_dir: The directory where events and weights are saved
                        checkpoint_path: the path to the last checkpoint file
                """                

                dir_names = next(os.walk(self.model_dir))[1]
                dir_names = sorted(dir_names)
                if not dir_names:
                        return None,None

                #pick last directory
                dir_name = os.path.join(self.model_dir,dir_names[-1])
                # Find the last checkpoint
                checkpoints = next(os.walk(dir_name))[2]
                checkpoints = sorted(checkpoints)
                if not checkpoints:
                        return dir_name,None
                checkpoint = os.path.join(dir_name, checkpoints[-1])
                return dir_name,checkpoint

        def load_weights(self,filepath):
                self.keras_model.load_weights(filepath)
                self.set_log_dir(filepath) 
                                                      
        def set_log_dir(self,model_path=None):
                """ Sets the model log directory and epoch counter.
                
                model_path: If None, or a format different from what this code uses
                        then set a new direcotyr and start epochs from 0. otherwise,
                        extract the log directory and the epoch counter from the file
                        name.
                """
                #Set date and epoch counter as if starting a new model
                self.epoch = 0
                now = datetime.datetime.now()

                #If we have a model path with date and epochs use them
                if model_path:
                        # A sample model path might look like:
                        # /path/to/logs/unet20180910T2215/unet_0001.h5
                        regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/unet\_(\d{4})\.h5"                                      
                        m = re.match(regex,model_path)
                        if m:
                                now = datetime.datetime(int(m.group(1)),int(m.group(2)),int(m.group(3)),
                                                        int(m.group(4)),int(m.group(5)))
                                self.epoch = int(m.group(6))

                self.log_dir = os.path.join(self.model_dir,"{}{:%Y%m%dT%H%M}".format("unet",now))
 
                if model_path==None:
                        os.chdir(self.model_dir)
                        os.mkdir(os.path.basename(self.log_dir))

                self.checkpoint_path = os.path.join(self.log_dir,"unet_*epoch*.h5")
                self.checkpoint_path = self.checkpoint_path.replace("*epoch*","{epoch:04d}")

        def train(self,train_dataset,val_dataset,learning_rate,epochs,steps_per_epoch,validation_steps,batch_size=1):

                train_generator = data_generator(dataset=train_dataset)
                val_generator   = data_generator(dataset=val_dataset)

                #callbacks
                callbacks = [
                keras.callbacks.TensorBoard(log_dir="../log",histogram_freq=0,
                                            write_graph=True,write_images=False),
                keras.callbacks.ModelCheckpoint(self.checkpoint_path,verbose=0,save_weights_only=True)
                ]

                #Train
                #log("\nStarting at epoch {}. LR={}\n".format(self.epoch,learning_rate))
                #log("Checkpoint Path: {}".format(self.checkpoint_path))
                adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-8, decay=0.0, amsgrad=False) 
                #self.keras_model.compile(optimizer=adam, loss='mean_squared_error',metrics=['mae'])
                self.keras_model.compile(optimizer=adam, loss=mean_square_loss, metrics=['mae'])
               

                if os.name is 'nt':
                        workers = 0
                else:
                        workers = multiprocessing.cpu_count()

                self.keras_model.fit_generator(
                        train_generator,
                        initial_epoch = self.epoch,
                        epochs = epochs,
                        steps_per_epoch = steps_per_epoch,
                        callbacks = callbacks,
                        validation_data = val_generator,
                        validation_steps =  validation_steps,
                        max_queue_size = 100,
                        workers = workers,
                        use_multiprocessing = True,
                )

                self.epoch = max(self.epoch, epochs)                        

        def detect(self,image_path,ratio):
               input_raw = rawpy.imread(image_path)
               img       = pack_raw(input_raw)*ratio
               H = (img.shape[0]/32)*32
               W = (img.shape[1]/32)*32
               img = img[:H,:W,:]
               denoised_image = self.keras_model.predict([img])
               return denoised_image                 

train_dataset = Dataset('train')
dev_dataset   = Dataset('dev')

#                                                      
model =  DenoiseUnet(mode='train', weights='new',model_dir="../log")
model.train(train_dataset=train_dataset,val_dataset=dev_dataset,learning_rate=0.00005,epochs=200,
            steps_per_epoch=5192,validation_steps=100,batch_size=1)  

#resume train                                                                                                            
weights_path = model.find_last()[1]
model.load_weights(weights_path)
model.train(train_dataset=train_dataset,val_dataset=dev_dataset,learning_rate=0.00005,epochs=200,
            steps_per_epoch=5192,validation_steps=100,batch_size=1)                                                   

                                                    
#for detect
model =  DenoiseUnet(mode='detect', weights='new',model_dir="../log")
weights_path = model.find_last()[1]
model.load_weights(weights_path)                                                      
denoised_image = model.detect(args.image)[0]
cv2.imwrite('denoised_image.jpg',denoised_image)                                                   
                                                      
                                                      
