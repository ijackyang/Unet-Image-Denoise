import glob
import numpy as np
import rawpy
import os

long_dir = '../dataset/Sony/long'
short_dir = '../dataset/Sony/short'

long_paths = glob.glob(os.path.join(long_dir,'*.ARW'))
short_paths = glob.glob(os.path.join(short_dir,'*.ARW'))

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


total_long = len(long_paths)
current_long = 0
for long_path in long_paths:
  print("Long Progress: %04d / %04d"%(current_long,total_long))
  current_long = current_long + 1
  long_img  = np.float32((rawpy.imread(long_path).postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16))/65535.0)
  long_fn   = os.path.basename(long_path)
  long_fn   = os.path.join(long_dir,(long_fn.split('.')[0] + '.npy')
  long_img.save(long_fn,long_img)
  os.remove(long_path)
  
total_short = len(short_paths)
current_short = 0
for short_path in short_paths:
  print("Long Progress: %04d / %04d"%(current_short,total_short))
  current_short = current_short + 1
  short_img = pack_raw(rawpy.imread(short_path))
  short_fn   = os.path.basename(short_path)
  short_fn   = os.path.join(short_dir,(short_fn.split('.')[0] + '.npy')
  long_img.save(short_fn,short_img)
  os.remove(short_path)
