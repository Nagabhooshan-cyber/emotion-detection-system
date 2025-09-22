# EMOTIONIQ — Real-Time Human Emotion Detection

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-orange)
![Keras](https://img.shields.io/badge/Keras-2.14-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green)
![Model Accuracy](https://img.shields.io/badge/Accuracy-91%25-brightgreen)
![Number of Classes](https://img.shields.io/badge/Classes-7-blue)


EMOTIONIQ is a real-time human emotion detection system built with **TensorFlow/Keras** and **OpenCV**.  
It classifies human emotions into 7 categories:  
`angry, disgust, fear, happy, sad, surprise, neutral`.

---

## 📂 Folder Structure
```
EMOTIONIQ/
│── code/
│ ├── train_model.py
│ ├── real_time_detect.py
│ ├── plot_graphs.py
│── dataset/
│ ├── train/
│ ├── test/
│── models/
│ ├── emotion_model.keras
│ ├── class_indices.json
│ ├── history.json
│ ├── accuracy_loss.png
│── snapshots/
│── requirements.txt
│── README.md
│── .gitignore
```
---

## 📥 Dataset Download
This project uses the FER 2013 dataset
 for training and testing.

🔽 Steps to Download:

1.Go to the dataset page on Kaggle:
👉 FER 2013 Dataset

2.Click the Download button (requires a free Kaggle account).

3.You will get a file named fer2013.zip.

4.Extract the contents of the zip file.

5.Place the extracted folders into your project structure so it looks like this:

```
EMOTIONIQ/
├── dataset/
│   ├── train/
│   └── test/
├── models/
├── code/
└── README.md

```
6.Once the dataset is placed correctly, you can train your model:
```
python train_model.py
```

7.For real-time emotion detection (using your webcam):
```
python real_time_detection.py
```
---

## ⚙️ Setup Instructions

### 1. Create Virtual Environment
```
python -m venv .venv
.venv\Scripts\Activate.ps1  
```
### 2. Install Dependencies
```
pip install -r requirements.txt
```
### 3. Train the Model
```
python code/train_model.py
```
Trains the CNN model on your dataset

Saves the model in models/emotion_model.keras

Saves class labels in models/class_indices.json



### 4. Run Real-Time Detection
```
python code/real_time_detect.py
```
Opens your webcam

Detects faces

Classifies emotions live

Saves snapshots in snapshots/

Press q to exit the webcam window.


---
### ✅ Requirements
---
Python 3.10+

TensorFlow

Keras

OpenCV

NumPy

---

### 🚀 Future Improvements
---
Improve accuracy using deeper CNN or transfer learning (ResNet, VGG)

Deploy as a web app using Streamlit

Add support for processing video files in addition to webcam

