#!/usr/bin/env python
'''
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
'''

import argparse
import numpy as np
import cv2
import glob
import h5py

from undistort_image import undistort_image

square_size = 1.0
pattern_size = (6, 9)

def show_frame(img, corners, index):
  # Draw and display the corners
  cv2.drawChessboardCorners(img, pattern_size, corners, True)
  #cv2.imshow('img',img)
  cv2.imwrite('/tmp/frame_%02d.png' % index, img)
  np.savetxt('/tmp/frame_%2d.txt' % index, corners.reshape(-1, 2))

def main(args):

  pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
  pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
  pattern_points *= square_size

  obj_points = []
  img_points = []
  h, w = 0, 0

  frames = []
  images = glob.glob('camera_cal/calibration*.jpg')
  for fname in images:
    frame = cv2.imread(fname)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if found:
      term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
      cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
      show_frame(gray, corners, len(frames))
      img_points.append(corners.reshape(-1, 2))
      obj_points.append(pattern_points)
      frames.append(frame)

  print('We got %s frames' % len(img_points))

  # calculate camera distortion
  rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None, flags = 0)

  print('camera_matrix', camera_matrix, 'dist_coefs', dist_coefs, '_rvecs', _rvecs, '_tvecs', _tvecs)

  with h5py.File(args.output, 'w') as f:
    f.create_dataset('camera_matrix', data=camera_matrix)
    f.create_dataset('dist_coefs', data=dist_coefs)


  # undisort the frames
  for i, frame in enumerate(frames):
    dst = undistort_image(frame, camera_matrix, dist_coefs)
    cv2.imwrite('/tmp/corrected_%02d.png' % i, dst)

  cv2.waitKey(500)
  cv2.destroyAllWindows()

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument("-o", "--output", type=str, default='result/calibration.h5',
    help="path to store calibration matrix")
  args = ap.parse_args()

  main(args)
