hand_create_csv.py - collecting hand gesture data for training
hand_gesture_data.csv - collected training data in *.csv format
hand_train.ipynb - Jupyter notebook file for model training
hand_gesture_model.h5 - trained model in *.h5 format
hand_gesture_model.tflite - trained modelin *.tflite format
hand_gesture_model_quantized.tflite - trained model converted to quantized model
hand_gesture_model_quantized_edgetpu.tflite - trained model for edge TPU 
hand_detect.py - run *.tflite, display the result as text at the image window (30fps)
hand_detect_h5.py - run *.h5, display the result as text at the image window (11fps)
hand_detect_quantized.py - run quantized *.tflite, display the result as text at the image window (30fps)
hand_detect_edgetpu.py - run hand_gesture_model_quantized_edgetpu.tflite 
hand_detect_move_ball.py - use gesture to move a ball and change its color (30fps)

