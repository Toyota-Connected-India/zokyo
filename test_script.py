import numpy as np
import cv2


image = cv2.imread("tests/images/0.png")
image = cv2.resize(image, (1280,720))
d_coef = (-2, 0, 0, 0, 0)
# get the height and the width of the image
h, w = image.shape[:2]
# compute its diagonal
f = (h ** 2 + w ** 2) ** 0.5
# set the image projective to carrtesian dimension
K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
# Generate new camera matrix from parameters
M, _ = cv2.getOptimalNewCameraMatrix(K, d_coef, (w, h), 0)
# Generate look-up tables for remapping the camera image
remap = cv2.initUndistortRectifyMap(K, d_coef, None, M, (w, h), 5)

map_x = remap[0]
map_y = remap[1]

print(map_x[633][721],map_y[633][721])
print(image[0][0])

# Remap the original image to a new image
mapped_image = cv2.remap(image, *remap, cv2.INTER_LINEAR)

image = cv2.circle(image, (721,633), 5, [255,0,0], -1)
mapped_image = cv2.circle(mapped_image, (int(map_y[633][721]), int(map_x[633][721])), 5, [255,0,0], -1)

cv2.imshow("normal", image)
cv2.imshow("mapped", mapped_image)
cv2.waitKey(0)