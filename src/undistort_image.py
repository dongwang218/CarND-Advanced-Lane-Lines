#!/usr/bin/env python
'''
* Apply a distortion correction to raw images.
'''

import argparse
import numpy as np
import cv2
import glob
import h5py


def undistort_image(img, camera_matrix, dist_coefs):
  h, w = img.shape[:2]
  alpha = 0 # let opencv crop
  newcameramtx, roi=cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w,h), alpha, (w,h))

  mapx,mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coefs,None,newcameramtx,(w,h),5)
  x,y,w,h = roi
  assert(w > 0 and h > 0)

  dst = cv2.remap(img, mapx, mapy,cv2.INTER_LINEAR)
  # crop the image
  #dst = dst[y:y+h, x:x+w, :]
  return dst

def main(args):

  with h5py.File(args.camera, 'r') as f:
    camera_matrix = np.array(f.get('camera_matrix'))
    dist_coefs = np.array(f.get('dist_coefs'))


  # undisort the frames
  for i, fname in enumerate(args.images):
    frame = cv2.imread(fname)
    dst = undistort_image(frame, camera_matrix, dist_coefs)
    cv2.imwrite('/tmp/corrected_%02d.png' % i, dst)

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument("-c", "--camera", type=str, default='result/calibration.h5',
    help="path to store calibration matrix")
  ap.add_argument("images", nargs='*',
                      help="images")
  args = ap.parse_args()

  main(args)
