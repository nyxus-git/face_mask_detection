import cv2
import numpy as np

image = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.imshow("Test Window", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
