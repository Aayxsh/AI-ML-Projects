import mediapipe as mp 
import numpy as np 
import uuid
import os
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#camera feed

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence = 0.8,min_tracking_confidence = 0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        
        #detections
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        result = hands.process(image)
        image.flags.writeable = True
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if result.multi_hand_landmarks:
            for num, hand in enumerate(result.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                         )
        
        cv2.imshow('Hand Detector', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()