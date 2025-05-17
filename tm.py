import cv2
import numpy as np
import os
import face_recognition
import pyttsx3
import time
import datetime


engine = pyttsx3.init()
voices = engine.getProperty('voices')

engine.setProperty('voice', voices[1].id) 
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

faces_folder = "C:/Users/tarek/OneDrive/Desktop/P&D/faces/"
known_face_encodings = []
known_face_names = []

for filename in os.listdir(faces_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(faces_folder, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])
        else:
            print(f"No face detected in {filename}")

print("Loaded faces:", known_face_names)

coins_folder = "C:/Users/tarek/OneDrive/Desktop/P&D/coins/"
known_coin_descriptors = []
known_coin_names = []
orb = cv2.ORB_create(nfeatures=1500, fastThreshold=5)

for filename in os.listdir(coins_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(coins_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        image = cv2.GaussianBlur(image, (5, 5), 0)

        keypoints, descriptors = orb.detectAndCompute(image, None)
        if descriptors is not None and len(keypoints) > 100:
            known_coin_descriptors.append(descriptors)
            known_coin_names.append(os.path.splitext(filename)[0])

print("Loaded coins:", known_coin_names)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

last_spoken_faces = set()
last_no_faces_time = 0
last_spoken_coin = ""
last_detect_time = 0
mute = False   

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=0.6)
        name = "Not Found"
        if True in matches:
            index = matches.index(True)
            name = known_face_names[index]
        face_names.append(name)

    current_faces = set(face_names)
    new_faces = current_faces - last_spoken_faces
    if not mute:
        for name in new_faces:
            engine.say(name)
            engine.runAndWait()
    last_spoken_faces = current_faces

    if not face_names:
        if time.time() - last_no_faces_time > 5 and not mute:

            engine.runAndWait()
            last_no_faces_time = time.time()

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (0, 0), fx=0.75, fy=0.75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    small = clahe.apply(small)
    blurred = cv2.GaussianBlur(small, (7, 7), 0)

    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 80,
                               param1=50, param2=15, minRadius=30, maxRadius=120)

    coin_name = "No coin detected"
    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest_circle = max(circles[0], key=lambda c: c[2])
        center = (int(largest_circle[0] / 0.75), int(largest_circle[1] / 0.75))
        radius = int(largest_circle[2] / 0.75)
        if radius > 30:
            cv2.circle(frame, center, radius, (255, 0, 0), 2)
            cv2.circle(frame, center, 3, (0, 0, 255), -1)

            if time.time() - last_detect_time > 0.5:
                keypoints_frame, descriptors_frame = orb.detectAndCompute(gray, None)
                last_detect_time = time.time()
            else:
                descriptors_frame = None

            if descriptors_frame is not None:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING)
                best_count, best_index = 0, -1
                for i, known_des in enumerate(known_coin_descriptors):
                    matches = bf.knnMatch(known_des, descriptors_frame, k=2)
                    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
                    if len(good) > best_count and len(good) > 15:
                        best_count = len(good)
                        best_index = i
                if best_index != -1:
                    coin_name = known_coin_names[best_index]

    if coin_name != last_spoken_coin and not mute:
        print("Coin:", coin_name)
        engine.say(f"This is {coin_name}")
        engine.runAndWait()
        last_spoken_coin = coin_name

    cv2.putText(frame, coin_name, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Face & Coin Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'): 
        current_time = datetime.datetime.now().strftime("%I:%M:%S %p")
        engine.say(f"The current time is {current_time}")
        engine.runAndWait()
    elif key == ord('d'):  
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        engine.say(f"Today is {current_date}")
        engine.runAndWait()
    elif key == ord('f'):  
        num_faces = len(face_locations)
        if num_faces == 0:
            engine.say("No faces detected")
        elif num_faces == 1:
            engine.say("One face detected")
        else:
            engine.say(f"{num_faces} faces detected")
        engine.runAndWait()
    elif key == ord('c'):     
        engine.say(f"This is {coin_name}")
        engine.runAndWait()
    elif key == ord('s'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"screenshot_{timestamp}.jpg", frame)
        engine.say("Screenshot saved")
        engine.runAndWait()
    elif key == ord('l'):   
        brightness = np.mean(gray)
        if brightness < 50:
            engine.say("Low lighting detected")
        elif brightness > 150:
            engine.say("Bright lighting detected")
        else:
            engine.say("Good lighting detected")
        engine.runAndWait()
    elif key == ord('m'):  
        mute = not mute
        engine.say("Mute on" if mute else "Mute off")
        engine.runAndWait()

cap.release()
cv2.destroyAllWindows()