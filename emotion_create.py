import os
import numpy as np
from tensorflow import keras
import sys
import io
import cv2

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

model = keras.models.load_model('D:/emotion/model/emotion_recognition_model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (48, 48))

    img_array = np.expand_dims(np.expand_dims(resized_frame, -1), 0)
    predictions = model.predict(img_array)
    emotion = emotion_labels[np.argmax(predictions)]

    cv2.putText(frame, emotion, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Emotion Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
