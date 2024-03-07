import cv2

tpath = 'HW0/data/bounding_box.txt'
impath = 'HW0/data/image.png'

img = cv2.imread(impath)

with open(tpath) as f:
    for line in f.readlines():
        if line[-1] == '\n':
            line = line[:-1]
        pos = []
        for s in line.split(' '):
            pos.append(int(s))
            
        cv2.rectangle(img, (pos[0], pos[1]), (pos[2], pos[3]), (0, 0, 255), 2)
            
# cv2.imshow('result', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite('HW0/hw0_112550013_1.png', img)