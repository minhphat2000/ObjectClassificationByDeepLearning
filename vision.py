import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model('model.h5')

# Define class names
class_names = ['Phat', 'Cup']

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from camera
    ret, frame = cap.read()

    # Preprocess image
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)

    # Add label to image
    cv2.putText(frame, class_names[class_idx], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show image
    cv2.imshow('Camera', frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close window
cap.release()
cv2.destroyAllWindows()
