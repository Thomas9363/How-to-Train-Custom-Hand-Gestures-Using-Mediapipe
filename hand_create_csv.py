# including count for each class and sorting based on classes
import cv2
import numpy as np
import mediapipe as mp
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# CSV file path
csv_file_path = 'hand_gesture_data.csv'

# Initialize the count dictionary
data_count = {i: 0 for i in range(26)}  # Assuming labels are 0-9

# Function to read the existing data from CSV
def read_csv_data(file_path):
    data = []
    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row and row[0].isdigit():
                data.append(row)
                label = int(row[0])
                if label in data_count:
                    data_count[label] += 1
    return data
# Function to normalize landmarks
def normalize_landmarks(landmarks):
    # Take the first landmark as the reference point (0, 0)
    base_x, base_y = landmarks[0].x, landmarks[0].y
    normalized = np.array([[lm.x - base_x, lm.y - base_y] for lm in landmarks])
    return normalized.flatten()

# Read the existing data
existing_data = read_csv_data(csv_file_path)

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

print("Press a, b, c ..... to record gesture data with the respective label.")
print("Press 'esc' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB as MediaPipe expects RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                cap.release()
                cv2.destroyAllWindows()
                exit()
            elif ord('a') <= key <= ord('z'):
                label = key - ord('a')
                print(f"Key pressed: {chr(key)}, Label: {label}")

                # Check for key  using 0~9
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord('q'):
            #     cap.release()
            #     cv2.destroyAllWindows()
            #     exit()
            # elif key in [ord(str(i)) for i in range(10)]:
            #     label = int(chr(key))

                # Normalize the landmarks
                normalized_landmarks = normalize_landmarks(hand_landmarks.landmark)

                # Create a row for CSV
                row = [label] + normalized_landmarks.tolist()

                # Append the row to the existing data
                existing_data.append(row)

                data_count[label] += 1
                if data_count !=0:
                    # print(f"Recorded gesture with label {label}. Data points per class: {data_count}")
                    print(f"Data points per class: {data_count}")


    # Display the frame
    cv2.imshow('Hand Gesture Data Collection', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Sort the data by class label
sorted_data = sorted(existing_data, key=lambda x: int(x[0]))

# Write the sorted data back to the CSV file
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write the header
    header = ['label']
    for i in range(21):
        header.append(f'x{i}')
        header.append(f'y{i}')
    csv_writer.writerow(header)
    # Write the sorted data
    csv_writer.writerows(sorted_data)

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
