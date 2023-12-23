
# --------------------------------------------------
# File Name: finger_counter.py
# --------------------------------------------------
# Date Completed: 11-12-2023
# --------------------------------------------------
# Description:
# This is a computer vision-based raised finger 
# counter program that utilizes the MediaPipe 
# library to identify hand landmarks and extract 
# relevant information to count the number of raised 
# fingers in a live webcam feed.
# --------------------------------------------------

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Unable to capture a frame from the camera.")
            continue

        image = cv2.flip(image, 1)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        fingerCount = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                handLabel = results.multi_handedness[handIndex].classification[0].label

                handLandmarks = []

                for landmarks in hand_landmarks.landmark:
                    handLandmarks.append([landmarks.x, landmarks.y])

                if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:       # Left Thumb
                    fingerCount += 1
                elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:    # Right Thumb
                    fingerCount += 1

                if handLandmarks[8][1] < handLandmarks[6][1]:       # Left & Right Index finger
                    fingerCount += 1
                if handLandmarks[12][1] < handLandmarks[10][1]:     # Left & Right Middle finger
                    fingerCount += 1
                if handLandmarks[16][1] < handLandmarks[14][1]:     # Left & Right Ring finger
                    fingerCount += 1
                if handLandmarks[20][1] < handLandmarks[18][1]:     # Left & Right Pinky
                    fingerCount += 1

                # Set the color to red and decrease the dot size
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

        height, width, _ = image.shape
        text_position_label = ((width - cv2.getTextSize("No. of raised fingers:", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]) // 2, 40)
        text_position_count = ((width - cv2.getTextSize(str(fingerCount), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0][0]) // 2, 95)

        cv2.putText(image, "No. of raised fingers:", text_position_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, str(fingerCount), text_position_count, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

        cv2.imshow("Raised Finger Counter", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
