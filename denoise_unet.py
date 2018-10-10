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


def smooth_l1_loss(denoised_image,target_image):
        diff = K.abs(denoised_image - target_image)
        less_than_one = K.cast(K.less(diff,1.0),"float32")
        loss = (less_than_one*0.5*diff**2) + (1-less_than_one)*(diff-0.5)
        return loss

def mean_square_loss(denoised_image,target_image):
        loss = tf.losses.mean_squared_error(denoised_image,target_image)
        return loss


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


class Dataset(object):

        def __init__(self,data_dir):
                self.base_path = data_dir
                self.image_ids = next(os.walk(data_dir))[1]
        
        def load_image(self,image_id):
                image_path = os.path.join(self.base_path,image_id,image_id+'.JPEG')
                noise_path = os.path.join(self.base_path,image_id,image_id+'_n.jpg')
                image = cv2.imread(image_path)
                noise = cv2.imread(noise_path)
                return image,noise

def data_generator_high_resolution(dataset,noise_num,batch_size,shuffle=True):
        b = 0
        image_ids   = np.copy(dataset.image_ids)
        while True:
                try:
                        image_id          = np.random.choice(image_ids)
                        image_subids      = random.sample(range(noise_num),batch_size)
                        target_img        = dataset.load_target_image(image_id)
                        batch_noise_image = np.zeros((batch_size,)+target_img.shape,dtype=target_img.dtype)
                        batch_target_image = np.zeros((batch_size,)+target_img.shape,dtype=target_img.dtype)
                        
                        while b<batch_size:
                                noise_img = dataset.load_noise_image(image_id,image_subids[b])
                                batch_noise_image[b,:,:,:] = noise_img
                                batch_target_image[b,:,:,:] = target_img
                                b = b+1
                        b = 0
                        inputs = [batch_noise_image]
                        outputs = [batch_target_image]
                        yield inputs,outputs
                except (GeneratorExit, KeyboardInterrupt):
                        raise


def data_generator(dataset,shuffle=True):
        image_ids   = np.copy(dataset.image_ids)
        while True:
                try:
                        image_id   = np.random.choice(image_ids)
                        target_img,noise_img = dataset.load_image(image_id)
                        batch_noise_image  = np.expand_dims(noise_img,axis=0)
                        batch_target_image = np.expand_dims(target_img,axis=0)
                        inputs = [batch_noise_image]
                        outputs = [batch_target_image]
                        if target_img.shape[0]<1024 and target_img.shape[1]<1024:
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

        def detect(self,image):
               img = np.zeros((1,image.shape[0],image.shape[1],image.shape[2]),dtype=np.uint8)
               img[0,:,:,:] = image
               denoised_image = self.keras_model.predict([img])
               return denoised_image                 


if __name__ == '__main__':
        import argparse

        parser = argparse.ArgumentParser(
                description="Train Unet to Denoise Image")
        parser.add_argument("command",
                            metavar="<command>",
                            help="'train' or 'detect'")
        parser.add_argument('--dataset',required=False,
                             metavar="/path/to/denoise/dataset",
                             help='Directory of denoising dataset')
        parser.add_argument('--weights',required=True,
                            metavar="/path/to/weights.h5",
                            help="Path to weights.h5 file or 'coco'")
        parser.add_argument('--image',required=False,
                             metavar="path or URL to image",
                             help='Image to denoise')
        args = parser.parse_args()

        if args.command=="train":
                assert args.dataset, "Argument --dataset is required fro training"
        elif args.command == "detect":
                assert args.image,"Provide --image to be denoised"

        print("Weights: ", args.weights)
        print("Dataset: ", args.dataset)
        
        model = DenoiseUnet(mode=args.command, weights=args.weights.lower(),model_dir="../log")
        
        if args.weights.lower() == "new":
                print("Model is initialized")
        elif args.weights.lower() == "last":
                weights_path = model.find_last()[1]
                model.load_weights(weights_path)

        train_dataset = Dataset('../dataset/train')
        dev_dataset   = Dataset('../dataset/dev')

        if args.command == "train":
                model.train(train_dataset=train_dataset,val_dataset=dev_dataset,learning_rate=0.00005,epochs=200,steps_per_epoch=5192,validation_steps=100,batch_size=1)
        elif args.command == "detect":
                img = cv2.imread(args.image)
                denoised_image = model.detect(img)[0]
                cv2.imwrite('denoised_image.jpg',denoised_image)






