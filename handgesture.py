

# import necessary packages
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)  # Change max_num_hands to 2
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')


# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Gesture Recognition Model Prediction (Updated to handle shape mismatch)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    className = ''

    # Post process the result
    if result.multi_hand_landmarks:
        for i, handslms in enumerate(result.multi_hand_landmarks):
            landmarks = []
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

            # Flatten landmarks and reshape (if required by your model)
            landmarks = np.array(landmarks).flatten().reshape(1, -1)  # Flatten to (1, 42 * 2) or (1, 84) if needed
            # Normalize landmarks if needed
            landmarks = landmarks / np.max(landmarks)

            # Debugging: Print the shape of landmarks
            print("Landmarks shape before prediction:", landmarks.shape)

            # Predict gesture
            prediction = model.predict(landmarks)  # Model expects (1, 84) or matching shape
            classID = np.argmax(prediction)
            className = classNames[classID]

            # Show the prediction on the frame for each hand
            cv2.putText(frame, f'Hand {i+1}: {className}', (10, 50 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0,0,255), 2, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
