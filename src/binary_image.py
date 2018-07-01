#!/usr/bin/env python
'''
* Use color transforms, gradients, etc., to create a thresholded binary image.
'''

import argparse
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

yellow_HSV_th_min = np.array([0, 100, 180])
yellow_HSV_th_max = np.array([50, 255, 255])

def thresh_frame_in_HSV(frame, min_values, max_values, verbose=False):
  HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

  min_th_ok = np.all(HSV > min_values, axis=2)
  max_th_ok = np.all(HSV < max_values, axis=2)

  out = np.logical_and(min_th_ok, max_th_ok)

  return out

def binarize(img, verbose=False):
  h, w = img.shape[:2]

  binary = np.zeros(shape=(h, w), dtype=np.uint8)

  # highlight yellow lines by threshold in HSV color space
  yellow_mask = thresh_frame_in_HSV(img, yellow_HSV_th_min, yellow_HSV_th_max, verbose=False)
  binary = np.logical_or(binary, yellow_mask)

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  eq_global = cv2.equalizeHist(gray)
  _, th = cv2.threshold(eq_global, thresh=250, maxval=255, type=cv2.THRESH_BINARY)
  white_mask = th
  binary = np.logical_or(binary, white_mask)

  edges = cv2.Canny(gray, 100, 200)
  binary = np.logical_or(binary, edges)

  # apply a light morphology to "fill the gaps" in the binary image
  kernel = np.ones((5, 5), np.uint8)
  closing = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

  if verbose:
    f, ax = plt.subplots(2, 3)
    f.set_facecolor('white')
    ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0, 0].set_title('input_frame')
    ax[0, 0].set_axis_off()
    ax[0, 0].set_axis_bgcolor('red')
    ax[0, 1].imshow(white_mask, cmap='gray')
    ax[0, 1].set_title('white mask')
    ax[0, 1].set_axis_off()

    ax[0, 2].imshow(yellow_mask, cmap='gray')
    ax[0, 2].set_title('yellow mask')
    ax[0, 2].set_axis_off()

    ax[1, 0].imshow(edges, cmap='gray')
    ax[1, 0].set_title('canny')
    ax[1, 0].set_axis_off()

    ax[1, 1].imshow(binary, cmap='gray')
    ax[1, 1].set_title('before closure')
    ax[1, 1].set_axis_off()

    ax[1, 2].imshow(closing, cmap='gray')
    ax[1, 2].set_title('after closure')
    ax[1, 2].set_axis_off()
    plt.show()

  return closing * 255

def main(args):
  for i, fname in enumerate(args.images):
    frame = cv2.imread(fname)
    dst = binarize(frame, True)
    cv2.imwrite('/tmp/binary_%02d.png' % i, dst)


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument("images", nargs='*',
                      help="images")
  args = ap.parse_args()

  main(args)
