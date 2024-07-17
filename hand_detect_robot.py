import cv2
import numpy as np
import mediapipe as mp
import time
from adafruit_servokit import ServoKit
import RPi.GPIO as GPIO
import platform

# initial servos & Led
kit = ServoKit(channels=16,address=0x40 )
LED_CHANNEL,PAN_CHANNEL,TILT_CHANNEL = 13, 14, 15
pan_initial_angle=90
tilt_initial_angle=90
kit.servo[PAN_CHANNEL].angle = pan_initial_angle
kit.servo[TILT_CHANNEL].angle = tilt_initial_angle
kit._pca.channels[LED_CHANNEL].duty_cycle = 65535
# Laser Beam & GPIO
laser = 12
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(laser, GPIO.OUT)
GPIO.output(laser, False)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(0, 255, 255), thickness=1)
handConStyle = mpDraw.DrawingSpec(color=(255, 255, 0), thickness=1)

# Check if running on Raspberry Pi
if platform.system() == 'Linux' and 'arm' in platform.machine():
    import tflite_runtime.interpreter as tflite
else:
    import tensorflow as tf

# Load the TFLite model and allocate tensors
model_path = 'hand_gesture_model.tflite'
if platform.system() == 'Linux' and 'arm' in platform.machine():
    interpreter = tflite.Interpreter(model_path=model_path)
else:
    interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Gesture mapping
gesture_names = ["Up", "Down", "Left", "Right", "Left Up", "Left Down", "Right Down", "Right Up", "Fire"]
width = 960
height = 540

# Function to normalize landmarks
def normalize_landmarks(landmarks):
    base_x, base_y = landmarks[0].x, landmarks[0].y
    normalized = np.array([[lm.x - base_x, lm.y - base_y] for lm in landmarks])
    return normalized.flatten()

def calculate_fps(prev_time, prev_fps):
    current_time = time.time()
    fps = 0.9 * prev_fps + 0.1 * (1 / (current_time - prev_time))
    return fps, current_time

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
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            normalized_landmarks = normalize_landmarks(hand_landmarks.landmark)
            input_data = np.array(normalized_landmarks, dtype=np.float32).reshape(input_details[0]['shape'])
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            predicted_class = np.argmax(output_data)
            gesture_name = gesture_names[predicted_class]
            print('Predicted gesture:', gesture_name)
# servo movement and toggle laser beam          
            if gesture_name == "Left":
                 kit.servo[PAN_CHANNEL].angle = kit.servo[PAN_CHANNEL].angle+1
                 GPIO.output(laser, False)
            elif gesture_name == "Right":
                 kit.servo[PAN_CHANNEL].angle = kit.servo[PAN_CHANNEL].angle-1
                 GPIO.output(laser, False)
            elif gesture_name == "Up":
                 kit.servo[TILT_CHANNEL].angle = kit.servo[TILT_CHANNEL].angle-1
                 GPIO.output(laser, False)
            elif gesture_name == "Down":
                 kit.servo[TILT_CHANNEL].angle = kit.servo[TILT_CHANNEL].angle + 1
                 GPIO.output(laser, False)
            elif gesture_name == "Right Down":
                 kit.servo[PAN_CHANNEL].angle = kit.servo[PAN_CHANNEL].angle-1
                 kit.servo[TILT_CHANNEL].angle = kit.servo[TILT_CHANNEL].angle + 1
                 GPIO.output(laser, False)
            elif gesture_name == "Right Up":
                 kit.servo[PAN_CHANNEL].angle = kit.servo[PAN_CHANNEL].angle-1
                 kit.servo[TILT_CHANNEL].angle = kit.servo[TILT_CHANNEL].angle - 1
                 GPIO.output(laser, False)
            elif gesture_name == "Left Up":
                 kit.servo[PAN_CHANNEL].angle = kit.servo[PAN_CHANNEL].angle+1
                 kit.servo[TILT_CHANNEL].angle = kit.servo[TILT_CHANNEL].angle - 1
                 GPIO.output(laser, False)
            elif gesture_name == "Left Down":
                 kit.servo[PAN_CHANNEL].angle = kit.servo[PAN_CHANNEL].angle+1
                 kit.servo[TILT_CHANNEL].angle = kit.servo[TILT_CHANNEL].angle + 1
                 GPIO.output(laser, False)
            elif gesture_name == "Fire":
                 GPIO.output(laser, True)
            else:
                 GPIO.output(laser, False)
# draw hand
            mpDraw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
            brect = calc_bounding_rect(frame, hand_landmarks)
            cv2.rectangle(frame, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 1)
            cv2.putText(frame, f'Gesture: {gesture_name}', (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    fps, prev_time = calculate_fps(prev_time, prev_fps)
    prev_fps = fps
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
# clean up
kit.servo[PAN_CHANNEL].angle = pan_initial_angle 
kit.servo[TILT_CHANNEL].angle = tilt_initial_angle
kit._pca.channels[LED_CHANNEL].duty_cycle = 0
cap.release()
cv2.destroyAllWindows()
