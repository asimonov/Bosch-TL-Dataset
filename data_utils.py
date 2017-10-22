#!/usr/bin/env python
"""
Data handling utils for training/classification

"""

import glob
#import sys
import os
import cv2
import numpy as np


def load_tl_extracts(data_dirs, desired_dim=(32,32)):
  """
  Loads *.png images of traffic lights from data_dirs directories.
  Resizes them to desired_dim.
  Extracts label name from filename (000007_redleft.png -> redleft)
  Uses linear interpolation.

  :param data_dirs: Paths to look for files
  :param desired_dim: tuple for desired image size

  :returns numpy arrays x and y, equally sized. x are images in OpenCV format (H, W, BGR), y are labels.
  """
  imgs = []
  labels = []
  for data_dir in data_dirs:
    for f in glob.glob(os.path.join(data_dir, '*.png')):
      fname = os.path.basename(f)
      img = cv2.imread(f)  # this loads in BGR order by default
      label = fname[7:-4]
      resized = cv2.resize(img, desired_dim, interpolation=cv2.INTER_LINEAR)
      imgs.append(resized)
      labels.append(label)
  return np.array(imgs), np.array(labels)


