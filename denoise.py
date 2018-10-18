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


# In[5]:


def lrelu(x):
    return K.maximum(x * 0.2, x)


# In[6]:


def unet():
        "unet for image reconstruction"
        P0 = x = KL.Input(shape=(None,None,3),name="u_net_input")
        x = KL.Conv2D(64, (3, 3), strides=(1, 1), name='conv1', use_bias=True,padding="same")(x)
        x = KL.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
        x = KL.Conv2D(64, (3, 3), strides=(2, 2), name='conv1a', use_bias=True,padding="same")(x)
        P1= x = KL.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)

        x = KL.Conv2D(64,(3,3),strides=(1,1),name='conv2',use_bias=True,padding="same")(x)
        x = KL.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
        x = KL.Conv2D(64,(3,3),strides=(2,2),name='conv2a',use_bias=True,padding="same")(x)
        P2= x = KL.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
        
        x = KL.Conv2D(128,(3,3),strides=(1,1),name='conv3',use_bias=True,padding="same")(x)
        x = KL.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
        x = KL.Conv2D(128,(3,3),strides=(2,2),name='conv3a',use_bias=True,padding="same")(x)
        P3= x = KL.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)

        x = KL.Conv2D(128,(3,3),strides=(1,1),name='conv4',use_bias=True,padding="same")(x)
        x = KL.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
        x = KL.Conv2D(128,(3,3),strides=(2,2),name='conv4a',use_bias=True,padding="same")(x)
        P4= x = KL.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)

        x = KL.Conv2D(256,(3,3),strides=(1,1),name='conv5',use_bias=True,padding="same")(x)
        x = KL.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
        x = KL.Conv2D(256,(3,3),strides=(2,2),name='conv5a',use_bias=True,padding="same")(x)
        P5= x = KL.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)

        x = KL.Deconvolution2D(nb_filter=128, nb_row=3, nb_col=3,subsample=(1, 1),name='deconv4',border_mode='same')(x)
        x = KL.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
        x = KL.Deconvolution2D(nb_filter=128, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv4a',border_mode='same')(x)
        x = KL.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
        C4= x = KL.Add()([P4,x])

        x = KL.Deconvolution2D(nb_filter=128, nb_row=3, nb_col=3,subsample=(1, 1),name='deconv3',border_mode='same')(x)
        x = KL.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
        x = KL.Deconvolution2D(nb_filter=128, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv3a',border_mode='same')(x)
        x = KL.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
        C3= x = KL.Add()([P3,x])


        x = KL.Deconvolution2D(nb_filter=64, nb_row=3, nb_col=3,subsample=(1, 1),name='deconv2',border_mode='same')(x)
        x = KL.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
        x = KL.Deconvolution2D(nb_filter=64, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv2a',border_mode='same')(x)
        x = KL.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
        C2= x = KL.Add()([P2,x])

        x = KL.Deconvolution2D(nb_filter=64, nb_row=3, nb_col=3,subsample=(1, 1),name='deconv1',border_mode='same')(x)
        x = KL.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
        x = KL.Deconvolution2D(nb_filter=64, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv1a',border_mode='same')(x)
        x = KL.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
        C1= x = KL.Add()([P1,x])

        x = KL.Deconvolution2D(nb_filter=64, nb_row=3, nb_col=3,subsample=(1, 1),name='deconv0',border_mode='same')(x)
        x = KL.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
        x = KL.Deconvolution2D(nb_filter=3, nb_row=3, nb_col=3,subsample=(2, 2),name='deconv0a',border_mode='same')(x)
        x = KL.Lambda(lambda t:keras.activations.relu(t,alpha=0.1))(x)
        C0= x = KL.Add()([P0,x])

        x = KL.Conv2D(3,(3,3),strides=(1,1),name='convr',use_bias=True,padding="same")(x)
        denoised_image = KL.Activation('linear')(x)

        model = KM.Model(inputs=P0,outputs=denoised_image)
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


# In[23]:


image_ids = [fn.split('.')[0][0:5] for fn in next(os.walk(os.path.join(data_dir,'long')))[2]]
glob.glob(os.path.join(data_dir,'short',image_ids[0]+'_*.ARW'))


# In[ ]:


class Dataset(object):

        def __init__(self,data_dir):
                self.base_path = data_dir
                self.image_ids = [fn.split('.')[0][0:5] for fn in next(os.walk(os.path.join(data_dir,'long')))[2]]
        
        def load_image(self,image_id):
                gt_path    = glob.glob(os.path.join(self.base_path,'long',image_id+'_*.ARW'))[0]
                gt_fn      = os.path.basename(gt_path)
                in_files   = glob.glob(os.path.join(self.base_path,'short',image_id+'_*.ARW'))
                in_path    = in_files[np.random.random_integers(0, len(in_files) - 1)]
                in_fn      = os.path.basename(in_path)
                
                in_expo    = float(in_fn[9:-5])
                gt_expo    = float(gt_fn[9:-5])
                ratio      = min(gt_expo/in_expo,300)
                
                
                image_path = os.path.join(self.base_path,image_id,image_id+'.jpg')
                noise_path = os.path.join(self.base_path,image_id,image_id+'_n.jpg')
                image = cv2.imread(image_path)
                noise = cv2.imread(noise_path)
                return image,noise

for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        st = time.time()
        cnt += 1

        if input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # crop
        H = input_images[str(ratio)[0:3]][ind].shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)

