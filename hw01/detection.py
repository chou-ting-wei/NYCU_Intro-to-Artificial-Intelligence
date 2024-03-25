import cv2
import matplotlib.pyplot as plt
import numpy as np

def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function ouput the specified area (parking space image) of the input frame according to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    left_front = (x1, y1)
    right_front = (x2, y2)
    left_bottom = (x3, y3)
    right_bottom = (x4, y4)
    src_pts = np.array([left_front, right_front, left_bottom, right_bottom]).astype(np.float32)
    dst_pts = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    projective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped


def detect(data_path, clf):
    """
    Please read detectData.txt to understand the format. 
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160) 
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec. 
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (ML_Models_pred.txt), the format is the same as Sample.txt.
    
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    with open(data_path, 'r') as file:
        parking_spaces = [list(map(int, line.strip().split())) for line in file.readlines()]

    cap = cv2.VideoCapture('HW1/For-student/ML_Models/data/detect/video.gif')

    predictions = []

    ret, frame = cap.read()
    first_frame = None
    frame_count = 0
    parking_spaces = parking_spaces[1:]
    while ret:
        frame_count += 1
        frame_predictions = []
        for space in parking_spaces:
            x1, y1, x2, y2, x3, y3, x4, y4 = space
            cropped_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, frame)

            processed_image = cv2.resize(cropped_image, (36, 16))
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            cls_img = np.array([processed_image.flatten()])

            is_car = clf.classify(cls_img)
            frame_predictions.append(is_car)

            if is_car:
                cv2.polylines(frame, [np.array([(x1, y1), (x2, y2), (x4, y4), (x3, y3)])], True, (0, 255, 0), 2)

        predictions.append(frame_predictions)

        if first_frame is None:
            first_frame = frame

        ret, frame = cap.read()

    cv2.imwrite('HW1/For-student/pred.png', cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    with open('HW1/For-student/pred.txt', 'w') as file:
        for i, frame_pred in enumerate(predictions):
            file.write(f'{" ".join(map(str, frame_pred))}\n')

    cap.release()
    # End your code (Part 4)
