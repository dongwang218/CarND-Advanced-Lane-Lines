## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this project, your goal is to write a software pipeline to identify the lane boundaries in a video.
![alt tag](https://raw.githubusercontent.com/dongwang218/CarND-Advanced-Lane-Lines/master/result/good.png)

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

Implementation
---
```
python src/calibration.py
python src/undistort_image.py test_images/straight_lines1.jpg
frameworkpython src/binary_image.py test_images/test5.jpg
frameworkpython src/birdeye_view.py test_images/test5.jpg
frameworkpython src/fit_lines.py test_images/test5.jpg
frameworkpython src/detect_lane.py -i test_images/test5.jpg
frameworkpython src/detect_lane.py -v project_video.mp4
```

Discussion
---
Using ROI and bird eye view to fit lane lines are interesting. It works good for mostly straightlines. But has trouble for curved lanes and non daylight images. In particular using histogram to identify the left and right line position could fail. Testing multiple rotated histogram may improve slightly. A learning based method (eg. SegNet) can be more robust.

![alt tag](https://raw.githubusercontent.com/dongwang218/CarND-Advanced-Lane-Lines/master/result/bad_fit.png)
