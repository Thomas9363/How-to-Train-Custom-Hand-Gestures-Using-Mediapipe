import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils  # Use drawing utility
handLmsStyle = mpDraw.DrawingSpec(color=(0, 255, 255), thickness=1)  # Define landmark style
handConStyle = mpDraw.DrawingSpec(color=(255, 255, 0), thickness=1)  # Define connection style

# Load the Keras model without compiling
model = load_model('hand_gesture_model.h5', compile=False)

# Gesture mapping
gesture_names = ["Up", "Down", "Left", "Right", "Left Up", "Left Down", "Right Down", "Right Up", "Fire"]
width = 640
height = 360

# Function to normalize landmarks
def normalize_landmarks(landmarks):
    # Take the first landmark as the reference point (0, 0)
    base_x, base_y = landmarks[0].x, landmarks[0].y
    normalized = np.array([[lm.x - base_x, lm.y - base_y] for lm in landmarks])
    return normalized.flatten()

# function calculate FPS
def calculate_fps(prev_time, prev_fps):
    current_time = time.time()
    fps = 0.9 * prev_fps + 0.1 * (1 / (current_time - prev_time))
    return fps, current_time

# function find bounding box
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv2.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

prev_time = time.time()
prev_fps = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (width, height))
    frame = cv2.flip(frame, 1)
    # Convert the frame to RGB as MediaPipe expects RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Normalize the landmarks
            normalized_landmarks = normalize_landmarks(hand_landmarks.landmark)

            # Prepare input data
            input_data = np.array(normalized_landmarks, dtype=np.float32).reshape(1, -1)

            # Run inference
            output_data = model.predict(input_data)

            # Interpret the results
            predicted_class = np.argmax(output_data)
            gesture_name = gesture_names[predicted_class]
            print('Predicted gesture:', gesture_name)

            # Draw the hand landmarks on the frame
            mpDraw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, handLmsStyle, handConStyle)  # Draw landmarks styles
            brect = calc_bounding_rect(frame, hand_landmarks)  # Calculate the bounding rectangle
            cv2.rectangle(frame, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 1)  # Draw the bounding rectangle

            # Display the predicted gesture on the frame
            cv2.putText(frame, f'Gesture: {gesture_name}', (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    fps, prev_time = calculate_fps(prev_time, prev_fps)  # Calculate and display FPS
    prev_fps = fps
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
