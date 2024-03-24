import cv2
import os

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

datasets = 'datasets'  # All the faces data will be present in this folder
sub_data = 'gopal'  # Sub-directory for your data

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = (130, 100)  # Defining the size of images

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)  # '0' is used for the default webcam, you can change it if you have other cameras

# The program loops until it has 100 images of the face.
count = 1
while count <= 100:
    (_, frame) = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite(f'{path}/{count}.png', face_resize)
        count += 1

    cv2.imshow('OpenCV', frame)
    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
