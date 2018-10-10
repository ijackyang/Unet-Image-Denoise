import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

base_path = "./dataset"

#for i in range(1,383):
#        print("Current Image Index: "+str(i)+"\n")
#        img     = cv2.imread("pic/"+str(i)+'.jpg')
#        img     = cv2.resize(img,(0,0),fx=0.25,fy=0.25)
#        new_w   = (img.shape[0]/32)*32
#        new_h   = (img.shape[1]/32)*32
#        print("original image shape: ")
#        print(img.shape)
#        img     = img[0:new_w,0:new_h,:]
#        print("new image shape: ")
#        print(img.shape)        
#        mean    = int(np.mean(img))
#        deltas  = np.random.randint(mean/8+1,mean*4,size=(10,))
#        np.random.shuffle(deltas)
#        path = os.path.join(base_path,str(i+700))
#        os.mkdir(path)       
#        for j,delta in enumerate(deltas):
#                noise   = np.random.normal(0,delta,img.shape)
#                syn_img = img + noise
#                syn_img = np.clip(syn_img,0,255)
#                syn_img = syn_img.astype(np.uint8)
#                cv2.imwrite(os.path.join(path,str(i+700)+".jpg"),img)
#                cv2.imwrite(os.path.join(path,str(i+700)+"_"+str(j)+".jpg"),syn_img)


dirs = next(os.walk('./pic'))[2]
i = 0

for dir in dirs:
        print(i)
        i = i+1
        img_name = os.path.splitext(dir)[0]
        img     = cv2.imread('pic/'+dir)


        new_w   = int((img.shape[0]/32)*32)
        new_h   = int((img.shape[1]/32)*32)
        img     = img[0:new_w,0:new_h,:]
        mean    = int(np.mean(img))
        delta  = np.random.randint(mean/8+1,mean*2)
        path = os.path.join(base_path,img_name)
        os.mkdir(path)       
        noise   = np.random.normal(0,delta,img.shape)
        syn_img = img + noise
        syn_img = np.clip(syn_img,0,255)
        syn_img = syn_img.astype(np.uint8)
        cv2.imwrite(os.path.join(path,dir),img)
        cv2.imwrite(os.path.join(path,img_name+"_n.jpg"),syn_img)

