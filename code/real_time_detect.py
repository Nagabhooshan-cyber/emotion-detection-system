import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Paths
model_path = "./models/emotion_model.keras"
classes_path = "./models/class_indices.json"


# Load model
model = load_model(model_path)

# Load class labels
with open(classes_path, "r") as f:
    class_indices = json.load(f)
emotion_labels = {v: k for k, v in class_indices.items()}  # invert dict

# Open webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48,48))
        roi_gray = roi_gray.astype("float")/255.0
        roi_gray = img_to_array(roi_gray)
        roi_gray = np.expand_dims(roi_gray, axis=0)

        preds = model.predict(roi_gray, verbose=0)[0]
        label = emotion_labels[np.argmax(preds)]
        score = np.max(preds)

        # Draw rectangle and label
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"{label} ({score:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # Save snapshot
        cv2.imwrite(f"C:/Users/NAGABHOOSHAN/OneDrive/Desktop/EMOTIONIQ/snapshots/{label}_{np.random.randint(1000)}.jpg", frame)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
