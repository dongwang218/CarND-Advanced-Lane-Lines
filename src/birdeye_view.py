#!/usr/bin/env python
'''
* Apply a perspective transform to rectify binary image ("birds-eye view").
'''

import argparse
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import h5py

from undistort_image import undistort_image
from binary_image import binarize

# assume the lane is about 30 meters long and 3.7 meters wide
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meteres per pixel in x dimension

def birdeye_view(img, verbose=False):
  h, w = img.shape[:2]

  #dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
  #src = np.array([[546, 460], [732, 460], [w, h-10], [0, h-10]], np.float32)
  src = np.float32([[585, 450], [203, 720], [1127, 720], [695, 450]])
  dst = np.float32([[320, 0], [320, 720], [960,720], [960, 0]])

  M = cv2.getPerspectiveTransform(src, dst)
  Minv = cv2.getPerspectiveTransform(dst, src)
  #print(M, Minv, np.dot(M, Minv))

  warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

  if verbose:
    f, axarray = plt.subplots(1, 2)
    f.set_facecolor('white')
    axarray[0].set_title('Before perspective transform')
    axarray[0].imshow(img, cmap='gray')
    for point in src:
        axarray[0].plot(point[0], point[1], '.', markersize=10)
    axarray[1].set_title('bird eye view')
    axarray[1].imshow(warped, cmap='gray')
    for axis in axarray:
        axis.set_axis_off()
    plt.show()

  return warped, Minv

def main(args):
  with h5py.File(args.camera, 'r') as f:
    camera_matrix = np.array(f.get('camera_matrix'))
    dist_coefs = np.array(f.get('dist_coefs'))

  for i, fname in enumerate(args.images):
    frame = cv2.imread(fname)
    dst = undistort_image(frame, camera_matrix, dist_coefs)
    binary = binarize(dst)
    view, _ = birdeye_view(binary, True)
    cv2.imwrite('/tmp/birdeyeview_%02d.png' % i, view)

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument("-c", "--camera", type=str, default='result/calibration.h5',
    help="path to store calibration matrix")
  ap.add_argument("images", nargs='*',
                      help="images")
  args = ap.parse_args()

  main(args)
