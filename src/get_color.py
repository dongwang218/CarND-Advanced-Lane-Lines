import os, sys
import cv2

image = cv2.imread(sys.argv[1])
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("image", image)

def tell_color(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONDOWN:
    print('bgr', image[y, x])
    print('hsv', hsv[y, x])

cv2.setMouseCallback("image", tell_color)

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	key = cv2.waitKey(1) & 0xFF

	# if the 'c' key is pressed, break from the loop
	if key == ord("q"):
		break
