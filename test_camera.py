import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print("Camera works!")
else:
    print("Camera not found")
cap.release()