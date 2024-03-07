import cv2
import numpy as np

impath = 'HW0/data/image.png'
img = cv2.imread(impath)
h, w, c = img.shape

# Translation
Mt = np.float32(([1, 0, 250], [0, 1, 100]))
img1 = cv2.warpAffine(img, Mt, (w, h))
cv2.imwrite('HW0/hw0_112550013_3/img1.png', img1)

# Rotation
cen = (w / 2, h / 2)
ang = 45
Mr = cv2.getRotationMatrix2D(cen, ang, 1.0)
img2 = cv2.warpAffine(img, Mr, (w, h))
cv2.imwrite('HW0/hw0_112550013_3/img2.png', img2)

# Flipping
img3 = cv2.flip(img, 0)
cv2.imwrite('HW0/hw0_112550013_3/img3.png', img3)

# Scaling
ssz = (640, 200)
img4 = cv2.resize(img, ssz)
cv2.imwrite('HW0/hw0_112550013_3/img4.png', img4)

# Cropping
cx = 200
cy = 100
cw = 300
ch = 400
img5 = img[cx:cx + cw, cy:cy + ch]
cv2.imwrite('HW0/hw0_112550013_3/img5.png', img5)

