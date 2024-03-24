# Import the necessary packages
import cv2
import argparse
import imutils
import face_recognition
import pickle
import numpy as np
from twilio.rest import Client
import serial
import time
port=serial.Serial("COM3",9600,timeout=0.1)

# Define the Twilio credentials and phone numbers
twilio_account_sid = 'ACcc0c0a3dbb38e843e863ad50d29693ec'
twilio_auth_token = 'a771c3b613ecf23d0e94d0887d268a52'
twilio_from_number = '+12408470852'
twilio_to_number = '+918939498280'

# Initialize the Twilio client
twilio_client = Client(twilio_account_sid, twilio_auth_token)

# Define the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", type=str, required=False, default="haarcascade_frontalface_default.xml",
                help="path to where the face cascade resides")
ap.add_argument("-e", "--encodings", type=str, required=False, default="encodings.pickle",
                help="path to serialized db of facial encodings")
ap.add_argument("-s", "--source", required=False, default=0,
                help="Use 0 for /dev/video0 or 'http://link.to/stream'")
ap.add_argument("-o", "--output", type=int, required=False, default=1,
                help="Show output")
ap.add_argument("-t", "--tolerance", type=float, required=False, default=0.4,
                help="How much distance between faces to consider it a match. Lower is more strict")
args = vars(ap.parse_args())

# Load the facial encodings and the face detector
data = pickle.loads(open(args["encodings"], "rb").read())
detector = cv2.CascadeClassifier(args["cascade"])

# Open the video source
vs = cv2.VideoCapture(args["source"])

tolerance = float(args["tolerance"])

while True:
    ret, frame = vs.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using the cascade classifier
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []  # Initialize names list
    
    for encoding in encodings:
        distances = face_recognition.face_distance(data["encodings"], encoding)
        minDistance = 1.0
        if len(distances) > 0:
            minDistance = min(distances)
        if minDistance < tolerance:
            idx = np.where(distances == minDistance)[0][0]
            name = data["names"][idx]
            print("Known person detected: " + name)
            port.write(str.encode("A"))
            time.sleep(2)
        else:
            name = "unknown person"
            print("Unknown person detected")
            port.write(str.encode("B"))
            time.sleep(2)
        names.append(name)  # Append the detected name to the names list
    
    # Loop through boxes and names to draw rectangles and labels
    for ((top, right, bottom, left), name) in zip(boxes, names):
        if name != "unknown person":
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Display a green boundary for known persons
        y = top - 15 if top - 15 > 15 else top + 15
        txt = name + " (" + "{:.2f}".format(minDistance) + ")"
        cv2.putText(frame, txt, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    
    if args["output"] == 1:
        cv2.imshow("Frame", frame)
    
    if "unknown person" in names:  # Check if any unknown person detected
        # Send a Twilio notification for unknown person
        twilio_client.messages.create(
            body="Unknown person detected",
            from_=twilio_from_number,
            to=twilio_to_number
        )
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.release()
