# How-to-Train-Custom-Hand-Gestures-Using-Mediapipe
<img src="/hand_gesture.GIF" alt="prototype" height="200">
This site provides a straightforward way to train computers using Mediapipe hand to detect your own hand gestures and use them to control a robotic device. It is divided to the following steps:

**Step 1: Data Collection**
*	Using MediaPipe to detect and extract landmark points from hand gestures in video frames.
*	Normalizing the landmark points to make the model training more effective.
*	Storing the extracted landmarks along with their corresponding gesture labels in a CSV file for training.
<img src="/data.png" alt="prototype" height="200">
***hand_create_csv.py:*** This script uses MediaPipe Hand to detect hand landmarks in video frames. Up to a total of 26 gestures can be extracted. You move your hand in front of the camera in different positions and angles as shown above. By pressing keyboard keys a to z, defined as class numbers 0-25, the x and y coordinates of the landmarks of your hand at that frame are extracted and normalized to point 0 (the wrist point). The class number and the flattened coordinates are stored as a row in a CSV file. The count of the dataset in each class is displayed when you show your hand and press a letter. The data are sorted during execution and stored in CSV file format. I have collected 60 sets of data for each gesture in different positions and angles, totaling 540 datasets. The resulting file is called “***hand_gesture_data.csv***” and is structured as shown above.

**Step 2: Model Training**
* Splitting the collected data into training, validation, and test sets.
* Creating a neural network model to classify gestures based on the extracted landmarks.
*	Using the Keras Sequential API to define the layers of the neural network, including input, hidden, and output layers.
*	Training the neural network on the preprocessed landmark data using TensorFlow’s training functionalities.
*	Validating the model on the validation set to tune hyperparameters and prevent overfitting.
<img src="/FCNN.png" alt="prototype" height="200">
***hand_train.ipynb:*** This notebook file starts the neural network training. The data from the CSV file is fed into a neural network model built with TensorFlow and Keras. Model layers are defined using the Sequential API. I am using a simple ReLU activation function. 
I am using Jupyter Notebook in PyCharm on my laptop to run the training. The training is fast and can be completed within a minute. After training, the model is saved. Several models in different formats are saved, including *.h5 and *.tflite files. The *.tflite is the trained model, which can be used on Windows OS or Raspberry Pi. Additionally, I convert the model to a TFLite model with quantization for later testing. At the end of this step, the following files are created: ***hand_gesture_model.h5, hand_gesture_model.tflite, and hand_gesture_model_quantized.tflite***.

**Step 3: Model Deployment and Inference**
*	Using the trained model to predict gestures in real-time based on new landmark data.
*	Porting to a Raspberry Pi to control a robotic device (optional).

***hand_detect.py:*** This script takes hand_gesture_model.tflite as input, performs inference, and displays the detected hand gesture as text on the screen. You press "Esc" to exit the script. Similarly, the other two scripts, ***hand_detect_h5.py*** and ***hand_detect_quantized.py***, are used to run ***hand_gesture_model.h5*** and ***hand_gesture_model_quantized.tflite***, respectively. These scripts can be run on both Windows and Raspberry Pi.

***hand_detect_move_ball.py***: This file contains an interactive graphic routine that allows you to use your hand gestures to roll a ball on the screen in different directions. It also changes the color of the ball when you show a "Fire" gesture. This script can also be run on both Windows and Raspberry Pi.
I have embedded the name of my gestures in the script:
gesture_names = ["Up", "Down", "Left", "Right", "Left Up", "Left Down", "Right Down", "Right Up", "Fire"]
If you train your own gestures with different name, you need to change them accordingly. Another thing to note is that you need to change cap = cv2.VideoCapture() to the number of camera you are using.

**Step 4: Model Deployment and Inference for Coral EdgeTPU**

To run the Edge TPU model on a Raspberry Pi, I have follow [the instructions](https://coral.ai/docs/accelerator/get-started/#requirements) to set up the Edge TPU runtime, the necessary software, and modify my script to use the Edge TPU. The script is ***hand_detect_edgetpu.py***. You also need to convert hand_gesture_model_quantized.tflite to ***hand_gesture_model_quantized_edgetpu.tflite*** using a web-based [Google Colab compiler](https://colab.research.google.com/github/google-coral/tutorials/blob/master/compile_for_edgetpu.ipynb).

**Step 5: Control a Robotic Device**

The robotic device contains a pan-tilt servo setup with two lasers mounted on top. ***hand_detect_robot.py*** is the script used to control the device. You can see the results in my [YouTube channel](https://www.youtube.com/watch?v=cxTbU-zHZtY). A more detailed description is located at my [instructables site](https://www.instructables.com/How-to-Train-Custom-Hand-Gestures-Using-Mediapipe/).


