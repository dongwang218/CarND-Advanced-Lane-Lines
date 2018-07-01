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
from birdeye_view import birdeye_view, ym_per_pix, xm_per_pix

window_width = 200

def fit_lines(binary, verbose=False):
  # find where it starts from near the camera
  h, w = binary.shape[:2]
  out_img = np.dstack((binary, binary, binary))

  counts = np.sum(binary[h//2:, :], axis=0)
  left_x = np.argmax(counts[:w//2])
  right_x = w//2 + np.argmax(counts[w//2:])

  n_window = 9
  window_height = h // n_window
  minpix = 50

  # y, x coordindates
  nonzero = binary.nonzero()
  nonzeroy = nonzero[0]
  nonzerox = nonzero[1]

  left_points = []
  right_points = []
  for index in range(n_window):
    center_y = h - index*window_height - window_height//2

    cv2.rectangle(out_img, (left_x-window_width//2, center_y-window_height//2),
                  (left_x+window_width//2, center_y+window_height//2), (0, 255, 0), 2)
    cv2.rectangle(out_img, (right_x-window_width//2, center_y-window_height//2),
                  (right_x+window_width//2, center_y+window_height//2), (0, 255, 0), 2)

    good_by_y = (np.abs(nonzeroy - center_y) <= window_height // 2)
    inside_left = ((np.abs(nonzerox - left_x) <= window_width // 2) & good_by_y).nonzero()[0]
    inside_right = ((np.abs(nonzerox - right_x) <= window_width // 2) & good_by_y).nonzero()[0]
    left_points.extend(inside_left)
    right_points.extend(inside_right)

    if inside_left.shape[0] > minpix:
      left_x = np.mean(nonzerox[inside_left]).astype(np.int)
    if inside_right.shape[0] > minpix:
      right_x = np.mean(nonzerox[inside_right]).astype(np.int)

  leftx = nonzerox[left_points]
  lefty = nonzeroy[left_points]
  rightx = nonzerox[right_points]
  righty = nonzeroy[right_points]

  if len(leftx) == 0:
    left_fit = None
  else:
    left_fit = np.polyfit(lefty, leftx, 2)

  if len(rightx) == 0:
    right_fit = None
  else:
    right_fit = np.polyfit(righty, rightx, 2)

  out_img[nonzeroy[left_points], nonzerox[left_points]] = [255, 0, 0]
  out_img[nonzeroy[right_points], nonzerox[right_points]] = [0, 0, 255]

  ploty = np.linspace(0, h-1, h).astype(np.int)
  if left_fit is not None:
    left_fitx = fitted_x(left_fit, ploty)
    out_img[ploty, left_fitx.astype(np.int)] = [128, 128, 0]
  if right_fit is not None:
    right_fitx = fitted_x(right_fit, ploty)
    out_img[ploty, right_fitx.astype(np.int)] = [255, 255, 0]

  if verbose:
    f, ax = plt.subplots(1, 2)
    f.set_facecolor('white')
    ax[0].imshow(binary, cmap='gray')
    ax[1].imshow(out_img)
    ax[1].set_xlim(0, 1280)
    ax[1].set_ylim(720, 0)
    plt.show()

  return left_fit, right_fit, out_img

def incremental_fit(left_fit, right_fit, binary):
  h, w = binary.shape[:2]
  ploty = np.linspace(0, h-1, h).astype(np.int)
  left_x = fitted_x(left_fit, ploty)
  right_x = fitted_x(right_fit, ploty)

  # y, x coordindates
  nonzero = binary.nonzero()
  nonzeroy = nonzero[0]
  nonzerox = nonzero[1]

  left_points = np.abs(nonzerox - fitted_x(left_fit, nonzeroy)) < window_width // 2
  right_points = np.abs(nonzerox - fitted_x(right_fit, nonzeroy)) < window_width // 2
  leftx = nonzerox[left_points]
  lefty = nonzeroy[left_points]
  rightx = nonzerox[right_points]
  righty = nonzeroy[right_points]

  if len(leftx) == 0:
    left_fit = None
  else:
    left_fit = np.polyfit(lefty, leftx, 2)

  if len(rightx) == 0:
    right_fit = None
  else:
    right_fit = np.polyfit(righty, rightx, 2)

  return left_fit, right_fit

def fitted_x(coef, y, max_x=1280):
  result = coef[0] * y**2 + coef[1] * y + coef[2]
  if isinstance(result, np.ndarray):
    return np.clip(result, 0, max_x-1)
  else:
    return min(max(0, result), max_x-1)

def curvature(left_fit, right_fit, binary):
  h, w = binary.shape[:2]
  ploty = np.linspace(0, h-1, h).astype(np.int)
  y_eval = h - 1
  left_curvature = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
  right_curvature = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
  center = ((fitted_x(left_fit, 720) + fitted_x(right_fit, 720) ) /2 - 640)*xm_per_pix

  return left_curvature, right_curvature, center

def main(args):
  with h5py.File(args.camera, 'r') as f:
    camera_matrix = np.array(f.get('camera_matrix'))
    dist_coefs = np.array(f.get('dist_coefs'))

  for i, fname in enumerate(args.images):
    frame = cv2.imread(fname)
    dst = undistort_image(frame, camera_matrix, dist_coefs)
    binary = binarize(dst)
    view, _ = birdeye_view(binary)
    left_fit, right_fit, out_img = fit_lines(view, True)
    cv2.imwrite('/tmp/lane_%02d.png' % i, out_img)

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument("-c", "--camera", type=str, default='result/calibration.h5',
    help="path to store calibration matrix")
  ap.add_argument("images", nargs='*',
                      help="images")
  args = ap.parse_args()

  main(args)
