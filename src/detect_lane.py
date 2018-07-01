#!/usr/bin/env python
'''
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
'''

import argparse
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import h5py
import traceback
import sys
import math

from undistort_image import undistort_image
from binary_image import binarize
from birdeye_view import birdeye_view, ym_per_pix, xm_per_pix
from fit_lines import fit_lines, incremental_fit, curvature, fitted_x


def detect_lane(undist, fit_state, verbose=False):
  binary = binarize(undist)
  view, Minv = birdeye_view(binary)
  if fit_state.get('left_fit') is None or fit_state.get('right_fit') is None:
    left_fit, right_fit, out_img = fit_lines(view)
  else:
    left_fit, right_fit = incremental_fit(fit_state.get('left_fit'), fit_state.get('right_fit'), view)

  if left_fit is not None and right_fit is not None:
    fit_state['left_fit'] = left_fit
    fit_state['right_fit'] = right_fit
    left_curv, right_curv, center_off = curvature(left_fit, right_fit, view)
    if abs(center_off) > 1: # todo a better sanity check
      print('bad detection')
      fit_state['left_fit'] = None
      fit_state['right_fit'] = None
      return None

    h, w = view.shape[:2]
    color_view = np.zeros((h, w, 3), dtype = np.uint8)
    ploty = np.linspace(0, h-1, h).astype(np.int)
    left_x = fitted_x(left_fit, ploty).astype(np.int)
    right_x = fitted_x(right_fit, ploty).astype(np.int)

    pts_left = np.vstack([left_x, ploty]).T
    pts_right = np.flipud(np.vstack([right_x, ploty]).T)
    pts = np.vstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_view, [pts], (255,215, 0))
    newview = cv2.warpPerspective(color_view, Minv, (w, h))
    img_out = cv2.addWeighted(undist, 1, newview, 0.3, 0)

    TextL = "Left curv: " + str(int(left_curv)) + " m"
    TextR = "Right curv: " + str(int(right_curv))+ " m"
    TextC = "Center offset: " + str(round( center_off,2)) + "m"
    fontScale=1
    thickness=2

    fontFace = cv2.FONT_HERSHEY_SIMPLEX


    cv2.putText(img_out, TextL, (130,40), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)
    cv2.putText(img_out, TextR, (130,70), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)
    cv2.putText(img_out, TextC, (130,100), fontFace, fontScale,(255,255,255), thickness,  lineType = cv2.LINE_AA)

    if verbose:
      f, ax = plt.subplots(1, 1)
      f.set_facecolor('white')
      ax.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
      ax.set_xlim(0, 1280)
      ax.set_ylim(720, 0)
      plt.show()

    return img_out

  else:
    return None

def main(args):
  with h5py.File(args.camera, 'r') as f:
    camera_matrix = np.array(f.get('camera_matrix'))
    dist_coefs = np.array(f.get('dist_coefs'))

  fit_state = {}
  if args.image:
    frame = cv2.imread(args.image)
    undist = undistort_image(frame, camera_matrix, dist_coefs)
    img_out = detect_lane(undist, fit_state, True)
    if img_out is not None:
      cv2.imwrite('/tmp/detected.png', img_out)
    else:
      print('no lane detected')

  if args.video:
    video = cv2.VideoCapture(args.video)
    if not video.isOpened():
        print "Could not open video"
        sys.exit()

    # get vcap property
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    #out_video = cv2.VideoWriter('/tmp/detected.mp4',
    #                            video.get(cv2.CAP_PROP_FOURCC),
    #                            int(video.get(cv2.CAP_PROP_FPS)),
    #                            (int(width), int(height)))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter('/tmp/output.avi', fourcc, video.get(cv2.CAP_PROP_FPS), (int(width), int(height)))

    count = 0
    while True:
      ok, frame = video.read()
      if not ok:
        print 'Cannot read video file'
        break
      undist = undistort_image(frame, camera_matrix, dist_coefs)
      try:
        img_out = detect_lane(undist, fit_state)
      except Exception as exc:
        traceback.print_exc()
        cv2.imwrite('/tmp/bad_detection_%02d.png' % count, frame)
        sys.exit(1)
      if img_out is None:
        print "failed to detect lane"
        img_out = undist
        count += 1
        cv2.imwrite('/tmp/bad_detection_%02d.png' % count, frame)

      out_video.write(img_out)
    out_video.release()

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument("-c", "--camera", type=str, default='result/calibration.h5',
    help="path to store calibration matrix")
  ap.add_argument('-i', '--image', type=str, help='input image to process')
  ap.add_argument('-v', '--video', type=str, help='input video to process')
  args = ap.parse_args()

  assert((args.image is None) != (args.video is None))
  main(args)
