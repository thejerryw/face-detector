# face_detector.py3 - detects face using opencv webcam display and shows rectangle around face

import cv2

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml" # builds full path to haarcascade which is built in object detection
a = cv2.CascadeClassifier(cascade_path) # loads path into object

b = cv2.VideoCapture(0) # opens webcam for live frame

while True: # infinite loop for camera detection
    c_rec, d_image = b.read() # c_rec is boolean [true if frame was read], d_image is actual frame
    e = cv2.cvtColor(d_image, cv2.COLOR_BGR2GRAY) # converts frame to grayscale [easier to process]
    f = a.detectMultiScale(e, 1.3, 6) # returns list of rectangles around face, scale factor, then minNeighbors or num. of positive detections to consider a face
 
    for (x1, y1, w1, h1) in f: 
        cv2.rectangle(d_image, (x1, y1), (x1+w1, y1+h1), (255, 0, 0), 5) # for every detected face, draws rectangle

    cv2.imshow('img', d_image) # displays updated image in window
    if cv2.waitKey(40) & 0xff == ord('q'): # waits 40 milliseconds for key press [press 'q']
        break # quits if pressed

b.release() # frees up webcam
cv2.destroyAllWindows() # destroys display window
